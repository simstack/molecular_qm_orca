from pathlib import Path

from simstack.core.node_runner import NodeRunner


class OrcaOutput(NodeRunner):
    def __init__(self, filename: Path, node_runner=None):
        
        super().__init__(node_runner)