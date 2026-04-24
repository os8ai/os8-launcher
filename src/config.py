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
    # Port as declared in config.yaml, before any user override from
    # ~/.config/os8-launcher/settings.yaml. Preserved so the Ports tab
    # can show "reset to default" and display both values side-by-side.
    default_port: int = 0


@dataclass
class ClientConfig:
    """A client entry from config.yaml."""
    name: str
    type: str  # "manifest" or "bridge"
    port: int | None = None
    manifest_path: str | None = None
    manifest: ManifestConfig | None = None
    default_port: int | None = None


@dataclass
class ResourcesConfig:
    """Soft-cap resource accounting for the Phase-2 resident pool.

    `memory_budget_gb` is the admission ceiling for sum(size_gb + kv_margin)
    across all running backends. On unified-memory hardware like DGX Spark
    this is a self-imposed policy, not a hardware limit — the OS will
    happily oversubscribe past it. 100 GB on a 128 GB Spark leaves 28 GB
    for the kernel and other processes.

    `kv_margin_gb` is a flat per-instance reservation for KV cache +
    launcher/docker overhead that model.size_gb doesn't capture.
    """
    memory_budget_gb: float = 100.0
    kv_margin_gb: float = 10.0
    auto_start_resident: bool = True
    auto_start_parallel: bool = False


@dataclass
class RoleConfig:
    """A named role (chat, coder, tts, …) and the (model, backend) it
    resolves to. The `resident:` list references role names, so swapping
    the model for a role doesn't require editing `resident:` too."""
    model: str
    backend: str | None = None


@dataclass
class Config:
    """Root configuration object."""
    models: dict[str, ModelConfig]
    backends: dict[str, BackendConfig]
    clients: dict[str, ClientConfig]
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    resident: list[str] = field(default_factory=list)
    roles: dict[str, RoleConfig] = field(default_factory=dict)

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
            default_port=port,
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
                default_port=port,
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
                default_port=port,
            )
    return clients


def _apply_port_overrides(config: "Config"):
    """Patch BackendConfig.port / ClientConfig.port from settings.yaml.

    Overrides are keyed by the name used in config.yaml. `default_port` keeps
    the value declared in config.yaml so callers (e.g. the /api/ports handler)
    can show both. Unknown names in the override file are ignored — they may
    belong to a backend/client removed from config.yaml.
    """
    from src.settings import get_port_overrides
    overrides = get_port_overrides()
    if not overrides:
        return
    for name, port in overrides.items():
        if name in config.backends:
            config.backends[name].port = port
        elif name in config.clients and config.clients[name].default_port is not None:
            # Only override clients that already had a port; bridge-only
            # entries with no port aren't meaningful to rewrite here.
            config.clients[name].port = port


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

    resources = _parse_resources(raw.get("resources") or {})
    resident = list(raw.get("resident") or [])
    roles = _parse_roles(raw.get("roles") or {})

    config = Config(
        models=models, backends=backends, clients=clients,
        resources=resources, resident=resident, roles=roles,
    )
    _validate_cross_references(config)
    _apply_port_overrides(config)

    return config


def _parse_resources(raw: dict) -> ResourcesConfig:
    """Parse the optional `resources:` section. Missing keys fall back
    to ResourcesConfig defaults so pre-Phase-2 config.yaml stays valid."""
    defaults = ResourcesConfig()
    return ResourcesConfig(
        memory_budget_gb=float(raw.get("memory_budget_gb", defaults.memory_budget_gb)),
        kv_margin_gb=float(raw.get("kv_margin_gb", defaults.kv_margin_gb)),
        auto_start_resident=bool(raw.get("auto_start_resident", defaults.auto_start_resident)),
        auto_start_parallel=bool(raw.get("auto_start_parallel", defaults.auto_start_parallel)),
    )


def _parse_roles(raw: dict) -> dict[str, RoleConfig]:
    """Parse the optional `roles:` section. Missing is fine — resident:
    entries for unknown roles are silently skipped at auto-start."""
    out: dict[str, RoleConfig] = {}
    for name, data in raw.items():
        if not isinstance(data, dict):
            continue
        model = data.get("model")
        if not model:
            continue
        out[name] = RoleConfig(model=model, backend=data.get("backend"))
    return out


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
                "size_gb": m.size_gb,
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
        "resident": list(config.resident),
        "roles": {
            name: {"model": r.model, "backend": r.backend}
            for name, r in config.roles.items()
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
