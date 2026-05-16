import os
import sys
import subprocess
import site

def build_exe():
    print("🚀 NithinVerse Build System Initializing...")
    
    # 1. Ensure PyInstaller is installed
    print("📦 Checking PyInstaller...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
    
    # 2. Find paths for complex libraries (MediaPipe & CustomTkinter)
    site_packages = site.getsitepackages()[0]
    
    mediapipe_path = os.path.join(site_packages, "mediapipe")
    ctk_path = os.path.join(site_packages, "customtkinter")
    
    print(f"🔍 Found MediaPipe at: {mediapipe_path}")
    print(f"🔍 Found CustomTkinter at: {ctk_path}")
    
    # 3. Construct the PyInstaller Command
    command = [
        "pyinstaller",
        "--noconfirm",
        "--onedir",           # Use --onedir instead of --onefile for heavy AI apps so it loads faster
        "--windowed",         # Hides the black terminal console
        "--name", "NithinVerse_V1",
        "--collect-all", "mediapipe",
        "--collect-all", "customtkinter",
        "--hidden-import", "speech_recognition",
        "--hidden-import", "pyaudio"
    ]
    
    # 4. Automatically attach the logo to the EXE if it exists in the main folder!
    icon_path = "icon.ico"
    if os.path.exists(icon_path):
        print(f"🎨 Found {icon_path}! Baking logo into the executable...")
        command.extend(["--icon", icon_path])
    else:
        print("⚠️ Warning: 'icon.ico' not found in the main folder. Using default exe icon.")
        
    command.append("app.py") # The name of your main script
    
    print("\n⚙️ Compiling EXE... (This will take 3-5 minutes, DO NOT CLOSE!)...")
    
    # Run the command
    subprocess.run(command, check=True)
    
    print("\n✅ BUILD SUCCESSFUL!")
    print("📁 Your game is located in the new 'dist/NithinVerse_V1' folder.")
    print("To share it with friends, ZIP the entire 'NithinVerse_V1' folder and send it to them!")

if __name__ == "__main__":
    build_exe()