import customtkinter as ctk
import time

class SplashScreen(ctk.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Center the splash screen
        window_width = 700
        window_height = 400
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = int((screen_width / 2) - (window_width / 2))
        y = int((screen_height / 2) - (window_height / 2))
        
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.overrideredirect(True) # Removes Windows borders
        self.configure(fg_color="#090a0f")
        self.attributes('-topmost', True)
        
        # UI Elements
        self.logo_label = ctk.CTkLabel(self, text="N_VIS", font=ctk.CTkFont(size=60, weight="bold"), text_color="#00f3ff")
        self.logo_label.pack(pady=(80, 10))
        
        # FIXED: Removed the invalid 'letter_spacing' argument here!
        self.title_label = ctk.CTkLabel(self, text="VISION GAMING ENGINE .02", font=ctk.CTkFont(size=20, weight="bold"), text_color="#ffffff")
        self.title_label.pack()
        
        self.dev_label = ctk.CTkLabel(self, text="Built by Nithin Gamers • NGI", font=ctk.CTkFont(size=12), text_color="#6b7280")
        self.dev_label.pack(pady=(5, 40))
        
        self.loading_text = ctk.CTkLabel(self, text="Calibrating AI Models...", font=ctk.CTkFont(size=12), text_color="#00f3ff")
        self.loading_text.pack(pady=(20, 5))
        
        self.progress_bar = ctk.CTkProgressBar(self, width=400, height=8, progress_color="#00f3ff", fg_color="#1f2937")
        self.progress_bar.pack()
        self.progress_bar.set(0)

    def show_duration(self, duration=3):
        steps = 100
        sleep_time = duration / steps
        for i in range(steps + 1):
            self.progress_bar.set(i / 100)
            if i == 30: self.loading_text.configure(text="Loading Pose Estimators...")
            if i == 60: self.loading_text.configure(text="Booting AR Overlays...")
            if i == 90: self.loading_text.configure(text="Engine Ready.")
            self.update()
            time.sleep(sleep_time)
        self.destroy()