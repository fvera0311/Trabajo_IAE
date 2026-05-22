"""
Módulo de integración Python-R
Elemento avanzado: Interoperabilidad entre lenguajes
"""
import subprocess
from pathlib import Path

class RPythonIntegration:
    def __init__(self):
        self.r_available = self.check_r_installed()
    
    def check_r_installed(self):
        try:
            # Quitamos shell=True. subprocess encontrará Rscript en el PATH automáticamente.
            result = subprocess.run(['Rscript', '--version'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception as e:
            print(f"Error comprobando R: {e}")
            return False
    
    def generate_r_visualizations(self, csv_path: str, output_dir: str = 'r_outputs'):
        if not self.r_available: 
            return False
            
        # Esta línea es perfecta, soluciona el problema de la carpeta temporal
        Path(output_dir).mkdir(exist_ok=True)
        r_script = Path('r_scripts/advanced_viz.R')
        
        if not r_script.exists(): 
            return False
        
        try:
            # Quitamos shell=True y mantenemos los 60 segundos
            result = subprocess.run(['Rscript', str(r_script), csv_path, output_dir],
                                    capture_output=True, text=True, timeout=60)
            
            # Si R falla, esto nos imprimirá el error exacto en los logs de Streamlit
            if result.returncode != 0:
                print(f"Error de R:\n{result.stderr}")
                
            return result.returncode == 0
        except Exception as e:
            print(f"Excepción ejecutando R: {e}")
            return False