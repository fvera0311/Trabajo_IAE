"""
Módulo de integración Python-R
Elemento avanzado: Interoperabilidad entre lenguajes
"""
import subprocess
import shutil
from pathlib import Path

class RPythonIntegration:
    def __init__(self):
        self.r_available = self.check_r_installed()
    
    def check_r_installed(self):
        try:
            # Añadimos shell=True
            result = subprocess.run(['Rscript', '--version'], capture_output=True, text=True, timeout=5, shell=True)
            return result.returncode == 0
        except:
            return False
    
    def generate_r_visualizations(self, csv_path: str, output_dir: str = 'r_outputs'):
        if not self.r_available: return False
        Path(output_dir).mkdir(exist_ok=True)
        r_script = Path('r_scripts/advanced_viz.R')
        
        if not r_script.exists(): return False
        
        try:
            # Añadimos shell=True y subimos a 60 segundos
            result = subprocess.run(['Rscript', str(r_script), csv_path, output_dir],
                                    capture_output=True, text=True, timeout=60, shell=True)
            return result.returncode == 0
        except:
            return False