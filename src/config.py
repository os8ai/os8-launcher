"""Parse, validate, and query os8-launcher configuration."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


class ConfigError(Exception):
    """Raised when configuration is invalid."""


@dataclass
class ManifestConfig:
    """A parsed manifest.yaml for a backend or client."""
    name: str
    install_type: str
    path: Path
    fields: dict = field(default_factory=dict)


@dataclass
class ModelConfig:
    """A model entry from config.yaml."""
    name: str
    source: str
    path: str
    format: str
    backends: list[str]
    default_backend: str
    downloaded: bool
    nim_image: str | None = None
    ollama_tag: str | None = None
    # Per-model vLLM image override. Set this when a model needs a different
    # vLLM build than the manifest's default (e.g. a nightly that shipped a
    # new model architecture or parser). Wired via the vllm manifest's
    # download.image_field, which also gates the auto-pull at start time.
    vllm_image: str | None = None
    size_gb: float | None = None
    backend_env: dict[str, str] = field(default_factory=dict)
    backend_args: str = ""
    allow_patterns: list[str] | None = None
    # Additional HF repos to pull into the SAME weights dir as the main source.
    # Each entry is {source: str, allow_patterns: list[str] | None}. Used when
    # a ComfyUI-style model (e.g. Flux Kontext) needs pieces scattered across
    # multiple repos — the Comfy-Org repack ships the transformer, the text
    # encoders live in comfyanonymous/flux_text_encoders, and the VAE comes
    # from the BFL repo. All three land in the one weights dir so the
    # ComfyUI bind-mount sees them under a single model name.
    extra_sources: list[dict] = field(default_factory=list)


@dataclass
class BackendConfig:
    """A serving backend entry from config.yaml."""
    name: str
    port: int
    manifest_path: str
    manifest: ManifestConfig | None = None


@dataclass
class ClientConfig:
    """A client entry from config.yaml."""
    name: str
    type: str  # "manifest" or "bridge"
    port: int | None = None
    manifest_path: str | None = None
    manifest: ManifestConfig | None = None


@dataclass
class Config:
    """Root configuration object."""
    models: dict[str, ModelConfig]
    backends: dict[str, BackendConfig]
    clients: dict[str, ClientConfig]

    def get_model(self, name: str) -> ModelConfig:
        if name not in self.models:
            raise ConfigError(f"Unknown model: {name}")
        return self.models[name]

    def get_backend(self, name: str) -> BackendConfig:
        if name not in self.backends:
            raise ConfigError(f"Unknown backend: {name}")
        return self.backends[name]

    def get_client(self, name: str) -> ClientConfig:
        if name not in self.clients:
            raise ConfigError(f"Unknown client: {name}")
        return self.clients[name]

    def get_backends_for_model(self, name: str) -> list[BackendConfig]:
        model = self.get_model(name)
        return [self.backends[b] for b in model.backends]


def _load_manifest(path: Path) -> ManifestConfig:
    """Load and parse a single manifest.yaml file."""
    if not path.exists():
        raise ConfigError(f"Manifest not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ConfigError(f"Manifest is not a valid YAML mapping: {path}")

    name = data.get("name")
    if not name:
        raise ConfigError(f"Manifest missing 'name': {path}")

    install_type = data.get("install_type")
    if not install_type:
        raise ConfigError(f"Manifest missing 'install_type': {path}")

    # Store all other fields for downstream use
    fields = {k: v for k, v in data.items() if k not in ("name", "install_type")}

    return ManifestConfig(name=name, install_type=install_type, path=path, fields=fields)


def _parse_models(raw: dict) -> dict[str, ModelConfig]:
    """Parse the models section of config.yaml."""
    models = {}
    for name, data in raw.items():
        required = ["source", "path", "format", "backends", "default_backend"]
        missing = [k for k in required if k not in data]
        if missing:
            raise ConfigError(f"Model '{name}' missing fields: {', '.join(missing)}")

        models[name] = ModelConfig(
            name=name,
            source=data["source"],
            path=data["path"],
            format=data["format"],
            backends=data["backends"],
            default_backend=data["default_backend"],
            downloaded=data.get("downloaded", False),
            nim_image=data.get("nim_image"),
            ollama_tag=data.get("ollama_tag"),
            vllm_image=data.get("vllm_image"),
            size_gb=data.get("size_gb"),
            backend_env=data.get("backend_env") or {},
            backend_args=data.get("backend_args", ""),
            allow_patterns=data.get("allow_patterns"),
            extra_sources=data.get("extra_sources") or [],
        )
    return models


def _parse_backends(raw: dict, repo_root: Path) -> dict[str, BackendConfig]:
    """Parse the backends section and load their manifests."""
    backends = {}
    for name, data in raw.items():
        manifest_path = data.get("manifest")
        if not manifest_path:
            raise ConfigError(f"Backend '{name}' missing 'manifest'")

        port = data.get("port", 8000)
        manifest = _load_manifest(repo_root / manifest_path)

        backends[name] = BackendConfig(
            name=name,
            port=port,
            manifest_path=manifest_path,
            manifest=manifest,
        )
    return backends


def _parse_clients(raw: dict, repo_root: Path) -> dict[str, ClientConfig]:
    """Parse the clients section and load their manifests."""
    clients = {}
    for name, data in raw.items():
        client_type = data.get("type", "manifest")
        port = data.get("port")

        if client_type == "bridge":
            clients[name] = ClientConfig(
                name=name,
                type="bridge",
                port=port,
            )
        else:
            manifest_path = data.get("manifest")
            if not manifest_path:
                raise ConfigError(f"Client '{name}' missing 'manifest'")

            manifest = _load_manifest(repo_root / manifest_path)
            clients[name] = ClientConfig(
                name=name,
                type="manifest",
                port=port,
                manifest_path=manifest_path,
                manifest=manifest,
            )
    return clients


def _validate_cross_references(config: Config):
    """Check that model backend references point to real backends."""
    for name, model in config.models.items():
        for backend_name in model.backends:
            if backend_name not in config.backends:
                raise ConfigError(
                    f"Model '{name}' references unknown backend '{backend_name}'"
                )
        if model.default_backend not in model.backends:
            raise ConfigError(
                f"Model '{name}' default_backend '{model.default_backend}' "
                f"is not in its backends list"
            )


def load_config(repo_root: str | Path) -> Config:
    """Load, parse, and validate the full configuration.

    Args:
        repo_root: Path to the repository root (where config.yaml lives).

    Returns:
        A validated Config object.
    """
    repo_root = Path(repo_root)
    config_path = repo_root / "config.yaml"

    if not config_path.exists():
        raise ConfigError(f"config.yaml not found at {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError("config.yaml is not a valid YAML mapping")

    for section in ("models", "backends", "clients"):
        if section not in raw:
            raise ConfigError(f"config.yaml missing '{section}' section")
        if not isinstance(raw[section], dict):
            raise ConfigError(f"config.yaml '{section}' must be a mapping")

    models = _parse_models(raw["models"])
    backends = _parse_backends(raw["backends"], repo_root)
    clients = _parse_clients(raw["clients"], repo_root)

    config = Config(models=models, backends=backends, clients=clients)
    _validate_cross_references(config)

    return config


def config_to_dict(config: Config) -> dict:
    """Serialize Config to a JSON-friendly dict."""
    return {
        "models": {
            name: {
                "source": m.source,
                "format": m.format,
                "path": m.path,
                "backends": m.backends,
                "default_backend": m.default_backend,
                "nim_image": m.nim_image,
            }
            for name, m in config.models.items()
        },
        "backends": {
            name: {
                "port": b.port,
                "install_type": b.manifest.install_type if b.manifest else None,
                "manifest_path": b.manifest_path,
            }
            for name, b in config.backends.items()
        },
        "clients": {
            name: {
                "type": c.type,
                "port": c.port,
                "install_type": c.manifest.install_type if c.manifest else None,
            }
            for name, c in config.clients.items()
        },
    }


def format_config(config: Config) -> str:
    """Format a Config object as a human-readable string."""
    lines = []

    lines.append("Models:")
    for name, m in config.models.items():
        status = "downloaded" if m.downloaded else "not downloaded"
        lines.append(f"  {name}")
        lines.append(f"    source:   {m.source}")
        lines.append(f"    format:   {m.format}")
        lines.append(f"    path:     {m.path}")
        lines.append(f"    status:   {status}")
        lines.append(f"    backends: {', '.join(m.backends)} (default: {m.default_backend})")
        if m.nim_image:
            lines.append(f"    nim_image: {m.nim_image}")

    lines.append("")
    lines.append("Backends:")
    for name, b in config.backends.items():
        lines.append(f"  {name}")
        lines.append(f"    type:     {b.manifest.install_type}")
        lines.append(f"    port:     {b.port}")
        lines.append(f"    manifest: {b.manifest_path}")

    lines.append("")
    lines.append("Clients:")
    for name, c in config.clients.items():
        lines.append(f"  {name}")
        if c.type == "bridge":
            lines.append(f"    type:     bridge")
        else:
            lines.append(f"    type:     {c.manifest.install_type}")
            lines.append(f"    manifest: {c.manifest_path}")
        if c.port:
            lines.append(f"    port:     {c.port}")

    return "\n".join(lines)
