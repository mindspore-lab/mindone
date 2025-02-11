from pathlib import Path

import yaml
from mkdocs.config.defaults import MkDocsConfig


def convert(toc, package, toc_file):
    toc_new = []
    for part in toc:
        if "sections" in part:
            toc_new.append({part["title"]: convert(part["sections"], package, toc_file)})
        elif "local" in part:
            toc_new.append({part["title"]: f'{package}/{part["local"]}.md'})
        else:
            raise ValueError(f"When parsing {toc_file}, got an unknown item: {part}")
    return toc_new


def on_config(config: MkDocsConfig) -> MkDocsConfig:
    for package, tab in [
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("peft", "PEFT"),
    ]:
        toc_file = f"{Path(__file__).parent.absolute()}/{package}/_toctree.yml"
        with open(toc_file, "r", encoding="utf-8") as f:
            toc = yaml.safe_load(f.read())

        config.nav.append(
            {tab: convert(toc, package, toc_file)},
        )

    return config
