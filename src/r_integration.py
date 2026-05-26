import subprocess
from pathlib import Path

class RPythonIntegration:
    def __init__(self):
        self.r_available = self.check_r_installed()
    
    def check_r_installed(self):
        try:
            result = subprocess.run(['Rscript', '--version'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception as e:
            print(f"Error comprobando R: {e}")
            return False
    
    def generate_r_visualizations(self, csv_path: str, output_dir: str = 'r_outputs'):
        if not self.r_available: 
            return False
            

        Path(output_dir).mkdir(exist_ok=True)
        r_script = Path('r_scripts/advanced_viz.R')
        
        if not r_script.exists(): 
            return False
        
        try:
            result = subprocess.run(['Rscript', str(r_script), csv_path, output_dir],
                                    capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"Error de R:\n{result.stderr}")
                
            return result.returncode == 0
        except Exception as e:
            print(f"Excepción ejecutando R: {e}")
            return False
