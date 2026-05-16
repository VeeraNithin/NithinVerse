import customtkinter as ctk
import tkinter as tk
import tkinter.messagebox as messagebox
import cv2
import mediapipe as mp
from PIL import Image, ImageTk
import math
import random
import time
import sys
import os
import traceback
from collections import deque
import numpy as np
import threading
import json

# --- Try to load Speech Recognition ---
try:
    import speech_recognition as sr
    HAS_VOICE = True
except ImportError:
    HAS_VOICE = False

try:
    from splash_screen import SplashScreen
    HAS_SPLASH = True
except ImportError:
    HAS_SPLASH = False

def resource_path(relative_path):
    try: base_path = sys._MEIPASS
    except Exception: base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ==========================================
# ASSET MANAGER & MEDIAPIPE SETUP
# ==========================================
os.makedirs("NithinVerse_Assets", exist_ok=True)

try:
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_selfie = mp.solutions.selfie_segmentation 
    mp_draw = mp.solutions.drawing_utils
except Exception as e:
    sys.exit(1)

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

def overlay_transparent(bg_img, img_to_overlay_t, x, y, overlay_size=None):
    """Blends a transparent PNG image perfectly onto the camera frame."""
    try:
        if overlay_size is not None:
            img_to_overlay_t = cv2.resize(img_to_overlay_t, overlay_size)
        
        bg_img = bg_img.copy()
        b, g, r, a = cv2.split(img_to_overlay_t)
        overlay_color = cv2.merge((b, g, r))
        mask = cv2.medianBlur(a, 5)
        h, w, _ = overlay_color.shape
        roi = bg_img[y:y+h, x:x+w]
        
        img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
        img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)
        bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)
        return bg_img
    except Exception:
        return bg_img # Return normal frame if bounds exceed or error occurs

# ==========================================
# TRUE DUAL-THREADED VISION ENGINE 
# ==========================================
class AsyncVisionThread:
    def __init__(self, src=0):
        self.src = src
        self.lock = threading.Lock() 
        self.running = True
        
        self.ret = False
        self.raw_frame = None
        self.processed_frame = None
        self.hand_res = None
        self.pose_res = None
        self.seg_mask = None
        self.game_mode = "MENU"
        
        self.cam_thread = threading.Thread(target=self._cam_loop, daemon=True)
        self.ai_thread = threading.Thread(target=self._ai_loop, daemon=True)
        
        self.cam_thread.start()
        self.ai_thread.start()

    def set_mode(self, mode):
        with self.lock:
            self.game_mode = mode

    def _cam_loop(self):
        def init_cam():
            if sys.platform == "win32":
                c = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
                c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                c.set(cv2.CAP_PROP_FPS, 30)
                c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                c = cv2.VideoCapture(self.src)
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
            return c

        cap = init_cam()
        fail_count = 0
        while self.running:
            ret, frame = cap.read()
            if ret:
                fail_count = 0
                with self.lock:
                    self.ret = ret
                    self.raw_frame = frame
            else:
                fail_count += 1
                time.sleep(0.01)
                if fail_count > 50:
                    cap.release()
                    time.sleep(0.5)
                    cap = init_cam()
                    fail_count = 0
        cap.release()

    def _ai_loop(self):
        hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        segmenter = mp_selfie.SelfieSegmentation(model_selection=1)
        
        dummy = np.zeros((600, 800, 3), dtype=np.uint8)
        hands.process(dummy)
        pose.process(dummy)
        segmenter.process(dummy)
        
        while self.running:
            raw = None
            with self.lock:
                if self.raw_frame is not None:
                    raw = self.raw_frame.copy()
                current_mode = self.game_mode
                
            if raw is not None:
                frame = cv2.flip(raw, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                h_res = hands.process(rgb_frame)
                p_res = None
                s_mask = None
                
                needs_pose = any(m in current_mode for m in ["AURA", "BEAM", "SHIELD", "DOMAIN", "AUTO", "BREATHING", "RASENGAN", "AMATERASU", "LIGHTNING", "DISTORTION", "BOSS", "COSPLAY"])
                if needs_pose: p_res = pose.process(rgb_frame)
                
                if "COSPLAY" in current_mode:
                    seg_res = segmenter.process(rgb_frame)
                    s_mask = seg_res.segmentation_mask
                
                with self.lock:
                    self.processed_frame = frame
                    self.hand_res = h_res
                    self.pose_res = p_res
                    self.seg_mask = s_mask
            
            time.sleep(0.015) 
            
        hands.close(); pose.close(); segmenter.close()

    def read(self):
        with self.lock:
            return self.ret, (self.processed_frame.copy() if self.processed_frame is not None else None), self.hand_res, self.pose_res, self.seg_mask

    def release(self):
        self.running = False
        if self.cam_thread.is_alive(): self.cam_thread.join(timeout=1.0)
        if self.ai_thread.is_alive(): self.ai_thread.join(timeout=1.0)


class NithinGamersApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        if HAS_SPLASH:
            splash = SplashScreen(self)
            splash.show_duration(3)
        
        # --- BRANDING ---
        self.title("NithinVerse Engine - By Nithin Anime Creations & Nithin Games")
        self.geometry("1400x900")
        self.minsize(1200, 800)
        self.configure(fg_color="#090a0f")
        
        self.is_running = True
        self.game_mode = "MENU" 
        self.vision_thread = None
        self.victory = False
        
        self.save_file = "nithinverse_profile.json"
        self.user_power = self._load_profile()
        self.loaded_images = {} # Cache for transparent PNGs
        
        self.all_anime_powers = self._generate_anime_powers()
        self.all_arcade_games = self._generate_arcade_games()
        self.all_cosplays = self._generate_cosplays()
        
        self.voice_mappings = {
            "flame": "hinokami", "shield": "domain expansion", "rasengan": "rasengan",
            "lightning": "chidori", "black_flame": "amaterasu"
        }
        self.active_voice_effect = None
        self.effect_timer = 0
        
        if HAS_VOICE:
            self.recognizer = sr.Recognizer()
            self.voice_thread = threading.Thread(target=self._listen_for_voice, daemon=True)
            self.voice_thread.start()
        
        # --- ENGINE STATES ---
        self.hand_trails = [deque(maxlen=15), deque(maxlen=15)]
        self.entities = [] 
        self.hit_markers = []
        
        self.global_score = 0
        self.lives = 5
        self.combo = 0
        self.combo_timer = 0
        self.difficulty_multiplier = 1.0
        
        self.ball = [{"x": 320, "y": 240, "dx": 10, "dy": 7, "radius": 15}]
        self.player_y = 240
        self.ai_paddle_y = 240
        self.boss = {}
        
        self.fx = {
            "impact_frames": 0, "show_speed_lines": False, "charge_level": 0, 
            "domain_radius": 0, "screen_shake": 0, "capture_timer": 0, "flash_text": ""
        }
        
        self._preload_assets()
        self._build_ui()
        self.after(500, self._start_camera)

    def _preload_assets(self):
        """Attempts to load real character PNGs from NithinVerse_Assets folder if user added them."""
        for file in os.listdir("NithinVerse_Assets"):
            if file.endswith(".png"):
                try:
                    img = cv2.imread(os.path.join("NithinVerse_Assets", file), cv2.IMREAD_UNCHANGED)
                    if img is not None and img.shape[2] == 4:
                        self.loaded_images[file.split(".")[0].upper()] = img
                except: pass

    # ==========================================
    # SYSTEM SAVES & GENERATORS
    # ==========================================
    def _load_profile(self):
        if os.path.exists(self.save_file):
            try:
                with open(self.save_file, "r") as f:
                    return json.load(f).get("power", None)
            except: pass
        return None

    def _save_profile(self, power_data):
        self.user_power = power_data
        with open(self.save_file, "w") as f:
            json.dump({"power": power_data}, f)

    def hex_to_bgr(self, hex_color):
        h = hex_color.lstrip('#')
        if len(h) != 6: return (0, 255, 255)
        return tuple(int(h[i:i+2], 16) for i in (4, 2, 0))

    def _generate_anime_powers(self):
        powers = [
            ("Flame Breathing", "🔥", "Leaves blazing fire trails.", "BREATHING_FIRE", "#ef4444"), 
            ("Water Breathing", "🌊", "Trails spawn falling water droplets.", "BREATHING_WATER", "#3b82f6"),
            ("Thunder Breathing", "⚡", "Trails spawn erratic lightning arcs.", "BREATHING_THUNDER", "#eab308"), 
            ("Wind Breathing", "🌪️", "Trails create swirling tornadoes.", "BREATHING_WIND", "#10b981"),
            ("Base Rasengan", "🌀", "Chakra sphere between hands.", "RASENGAN_BASE", "#38bdf8"), 
            ("Giant Rasengan", "🔵", "Massive crushing sphere.", "RASENGAN_GIANT", "#0ea5e9"),
            ("Malevolent Shrine", "⛩️", "Cross arms. Slashes everything.", "DOMAIN_SHRINE", "#dc2626"), 
            ("Infinite Void", "🌌", "Cross arms. Traps in starry space.", "DOMAIN_VOID", "#1e3a8a"),
            ("Kamehameha", "💥", "Classic blue destruction beam.", "BEAM_KAME", "#06b6d4"), 
            ("Final Flash", "⚡", "Blinding yellow dual-palm blast.", "BEAM_FINAL", "#facc15"),
            ("Sorcerer Shield", "🛡️", "Rotating runic mandala.", "SHIELD_BASE", "#f59e0b"), 
            ("Amaterasu", "🔥", "Point to spawn undying black fire.", "AMATERASU", "#a855f7"),
            ("Chidori", "⚡", "Palm thrust shoots wild lightning.", "LIGHTNING", "#3b82f6"),
            ("AI Auto-Mage", "🧠", "AI automatically chooses your power.", "AUTO", "#facc15")
        ]
        for i in range(1, 37): powers.append((f"Secret Art Vol. {i}", "✨", "Hidden NithinVerse Tech", f"BREATHING_SECRET_{i}", "#6366f1"))
        return powers

    def _generate_arcade_games(self):
        games = [
            ("Classic Slasher", "⚔️", "Slice falling orbs.", "SLASHER_NORMAL", "#f43f5e"),
            ("Bomb Slasher", "💣", "Slice fruits, DO NOT slice bombs.", "SLASHER_BOMB", "#ef4444"),
            ("Coin Catcher", "💰", "Keep palm open to catch coins.", "CATCHER_COIN", "#eab308"),
            ("Space Invaders", "👾", "Close fist to shoot lasers.", "INVADERS_NORMAL", "#3b82f6"),
            ("Shuriken Master", "🥷", "Flick wrist up to throw.", "SHURIKEN_NORMAL", "#6366f1"),
            ("Cyber Pong", "🏓", "Standard AI Paddle Pong.", "PONG_NORMAL", "#10b981"),
            ("Multi-Ball Pong", "🎱", "Survive against 3 balls at once.", "PONG_MULTI", "#facc15"),
            ("Meteor Smash", "🥊", "Punch targets quickly.", "METEOR_NORMAL", "#d946ef"),
            ("Zombie Boxing", "🧟", "Punch zombies moving toward you.", "METEOR_ZOMBIE", "#84cc16"),
            ("Flappy Hand", "🦅", "Move hand up/down to dodge pipes.", "FLAPPY_NORMAL", "#eab308"),
        ]
        modes = ["Neon", "Ice", "Shadow", "Plasma", "Chaos", "Quantum", "Turbo", "Cosmic"]
        for m in modes:
            games.append((f"{m} Slasher", "⚔️", f"Slasher with {m} physics.", f"SLASHER_{m.upper()}", "#f43f5e"))
            games.append((f"{m} Invaders", "👾", f"Invaders with {m} physics.", f"INVADERS_{m.upper()}", "#3b82f6"))
            games.append((f"{m} Pong", "🏓", f"Pong with {m} physics.", f"PONG_{m.upper()}", "#10b981"))
            games.append((f"{m} Smash", "🥊", f"Brawler with {m} physics.", f"METEOR_{m.upper()}", "#d946ef"))
            games.append((f"{m} Shuriken", "🥷", f"Thrower with {m} physics.", f"SHURIKEN_{m.upper()}", "#6366f1"))
        return games

    def _generate_cosplays(self):
        cosplays = [
            ("Akatsuki Member", "☁️", "Black cloak, red clouds, Rain Village background.", "COSPLAY_AKATSUKI", "#dc2626"),
            ("Demon Slayer", "🗡️", "Green/Black Checkered Haori, Bamboo Forest.", "COSPLAY_TANJIRO", "#10b981"),
            ("Super Saiyan", "🐉", "Orange Gi, Speed Lines, Namek Aura.", "COSPLAY_GOKU", "#f59e0b"),
            ("Jujutsu Sorcerer", "🕶️", "Black high collar, Blindfold, Shibuya background.", "COSPLAY_GOJO", "#3b82f6"),
            ("Cyberpunk Ninja", "🤖", "Neon grid clothes, Synthwave City.", "COSPLAY_CYBER", "#d946ef")
        ]
        for i in range(1, 46): cosplays.append((f"Virtual Skin {i}", "👕", "Procedurally generated anime skin.", f"COSPLAY_RANDOM_{i}", f"#{random.randint(100000, 999999)}"))
        return cosplays

    # ==========================================
    # 1. UI ANIMATIONS & MASSIVE MENU SYSTEM
    # ==========================================
    def slide_transition(self, frame_out, frame_in):
        """Smooth kinetic UI animation to transition between screens."""
        def step(current_step, total_steps):
            if current_step <= total_steps:
                progress = current_step / total_steps
                ease = 1 - math.pow(1 - progress, 3) # Cubic ease-out
                
                # Slide out left
                if frame_out: frame_out.place(relx=0.5 - ease, rely=0.5, anchor="center")
                # Slide in from right
                if frame_in: frame_in.place(relx=1.5 - ease, rely=0.5, anchor="center")
                
                self.after(16, lambda: step(current_step + 1, total_steps))
            else:
                if frame_out: frame_out.place_forget()
                if frame_in: frame_in.place(relx=0.5, rely=0.5, anchor="center")
                
        if frame_in: frame_in.place(relx=1.5, rely=0.5, anchor="center", relwidth=0.95, relheight=0.9)
        step(0, 20) 

    def _build_ui(self):
        nav = ctk.CTkFrame(self, height=65, corner_radius=0, fg_color="#111827", border_color="#374151", border_width=1)
        nav.pack(side="top", fill="x")
        
        ctk.CTkLabel(nav, text="N_VISION STUDIO", font=ctk.CTkFont(size=26, weight="bold"), text_color="#00f3ff").pack(side="left", padx=20, pady=15)
        ctk.CTkLabel(nav, text="The NithinVerse Engine", font=ctk.CTkFont(size=14, slant="italic", weight="bold"), text_color="#f59e0b").pack(side="left", pady=20)
        
        self.btn_menu = ctk.CTkButton(nav, text="🔙 Return to Hub", fg_color="#ef4444", hover_color="#b91c1c", font=ctk.CTkFont(weight="bold"), command=self.go_to_menu)
        self.btn_menu.pack(side="right", padx=20, pady=15)
        self.btn_menu.pack_forget() 
        
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True)
        
        self.menu_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.menu_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=1.0, relheight=1.0)
        
        self.tabview = ctk.CTkTabview(self.menu_frame, fg_color="#1f2937", segmented_button_selected_color="#3b82f6", segmented_button_unselected_color="#111827")
        self.tabview.pack(fill="both", expand=True, padx=20, pady=20)
        
        tab_cosplay = self.tabview.add("🎭 AR Cosplay Booth")
        tab_box = self.tabview.add("🎁 My Power Box")
        tab_boss = self.tabview.add("👹 Epic Boss Fights")
        tab_anime = self.tabview.add("🌌 NithinVerse Animes (50+)")
        tab_vision = self.tabview.add("🎮 Nithin Games (50+)")
        tab_manual = self.tabview.add("📖 Manual")
        tab_contact = self.tabview.add("📞 Contact NGI")
        
        self._build_cosplay_tab(tab_cosplay)
        self._build_powerbox_tab(tab_box)
        self._build_boss_tab(tab_boss)
        self._build_grid_tab(tab_anime, self.all_anime_powers, "Powered by Nithin Anime Creations", "#a855f7")
        self._build_grid_tab(tab_vision, self.all_arcade_games, "Powered by Nithin Games", "#10b981")
        self._build_manual_tab(tab_manual)
        self._build_contact_tab(tab_contact)

        self.game_frame = ctk.CTkFrame(self.main_container, fg_color="#000000", corner_radius=10)
        self.video_label = tk.Label(self.game_frame, bg="black")
        self.video_label.pack(fill="both", expand=True, padx=5, pady=5)

    def _build_manual_tab(self, parent):
        scroll_frame = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(scroll_frame, text="📖 Official NithinVerse Manual", font=ctk.CTkFont(size=32, weight="bold"), text_color="#00f3ff").pack(pady=(0, 20))
        
        instructions = """
        WELCOME TO THE NITHINVERSE ENGINE!

        1. 🎭 AR Cosplay Engine:
        Select an outfit. The AI removes your room's background and procedurally draws anime gear onto your body. Add real transparent PNGs to 'NithinVerse_Assets' to upgrade the graphics automatically!

        2. 🎁 Power Box (Awakening):
        Go to the Power Box to awaken your permanent soul power. This power represents your elemental aura in Boss Fights.

        3. 👹 Epic Boss Battles (Real Characters):
        The AI generates massive Boss avatars. 
        - Defend: Open both palms facing the camera to spawn your shield and reflect lasers.
        - Attack (Beam): Point one finger forward to shoot.
        - Attack (Smash): Bring wrists together to detonate a sphere.

        4. 🎮 Nithin Games (Infinite Arcades):
        50+ Vision Games using your hands. Combo multipliers go infinite! Lose 5 lives and it's GAME OVER.
        """
        label = ctk.CTkLabel(scroll_frame, text=instructions, font=ctk.CTkFont(size=15), text_color="#e2e8f0", justify="left", wraplength=1000)
        label.pack(anchor="w")

    def _build_contact_tab(self, parent):
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(expand=True, fill="both")
        
        ctk.CTkLabel(container, text="📞 Contact The Creator", font=ctk.CTkFont(size=36, weight="bold"), text_color="#facc15").pack(pady=(60, 10))
        ctk.CTkLabel(container, text="Nithin Anime Creations & Nithin Games", font=ctk.CTkFont(size=18, slant="italic"), text_color="#a855f7").pack(pady=(0, 40))
        
        card = ctk.CTkFrame(container, fg_color="#111827", corner_radius=15, border_width=2, border_color="#3b82f6")
        card.pack(ipadx=40, ipady=40)
        
        ctk.CTkLabel(card, text="Support & Business Inquiries", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)
        ctk.CTkLabel(card, text="Email: nithingroup1@gmail.com", text_color="#9ca3af", font=ctk.CTkFont(size=16)).pack(pady=5)
        ctk.CTkLabel(card, text="github:https://github.com/VeeraNithin ", text_color="#9ca3af", font=ctk.CTkFont(size=16)).pack(pady=5)
        ctk.CTkLabel(card, text="Discord: NGI Studio Hub", text_color="#9ca3af", font=ctk.CTkFont(size=16)).pack(pady=5)

    def _build_cosplay_tab(self, parent):
        scroll_frame = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True)
        
        ctk.CTkLabel(scroll_frame, text="🎭 AR Cosplay Simulator", font=ctk.CTkFont(size=32, weight="bold"), text_color="#f43f5e").pack(pady=10)
        ctk.CTkLabel(scroll_frame, text="Teleport into Anime worlds. Pro Tip: Add 'akatsuki.png' or 'goku.png' to 'NithinVerse_Assets' for real textures!", text_color="#9ca3af", font=ctk.CTkFont(size=14)).pack(pady=(0,20))
        
        grid = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        grid.pack(expand=True)
        
        for i, (title, icon, desc, mode, color) in enumerate(self.all_cosplays):
            card = ctk.CTkFrame(grid, fg_color="#111827", corner_radius=15, border_width=2, border_color=color)
            card.grid(row=i//4, column=i%4, padx=15, pady=15, ipadx=10, ipady=10, sticky="nsew")
            ctk.CTkLabel(card, text=f"{icon} {title}", font=ctk.CTkFont(size=18, weight="bold"), text_color=color).pack(pady=5)
            ctk.CTkLabel(card, text=desc, text_color="#9ca3af", font=ctk.CTkFont(size=11), wraplength=180).pack(pady=5)
            ctk.CTkButton(card, text="WEAR COSPLAY", fg_color=color, hover_color="#0f172a", font=ctk.CTkFont(weight="bold"), command=lambda m=mode: self.start_game(m)).pack(pady=10)

    def _build_powerbox_tab(self, parent):
        self.box_container = ctk.CTkFrame(parent, fg_color="transparent")
        self.box_container.pack(expand=True, fill="both")
        
        ctk.CTkLabel(self.box_container, text="NithinVerse Power Awakening", font=ctk.CTkFont(size=36, weight="bold"), text_color="#facc15").pack(pady=(40, 10))
        ctk.CTkLabel(self.box_container, text="Every user is born with one unique ability in the NithinVerse.", text_color="#9ca3af").pack(pady=(0, 40))
        
        self.power_display = ctk.CTkLabel(self.box_container, text="", font=ctk.CTkFont(size=28, weight="bold"))
        self.power_display.pack(pady=20)
        
        self.btn_awaken = ctk.CTkButton(self.box_container, text="AWAKEN MY POWER", font=ctk.CTkFont(size=20, weight="bold"), height=60, fg_color="#ef4444", hover_color="#b91c1c", command=self._trigger_awakening)
        self.btn_awaken.pack(pady=20)
        
        self.btn_cast_power = ctk.CTkButton(self.box_container, text="CAST MY POWER", font=ctk.CTkFont(size=20, weight="bold"), height=60, fg_color="#3b82f6", hover_color="#1d4ed8", command=self._cast_my_power)
        
        self._refresh_powerbox_ui()

    def _refresh_powerbox_ui(self):
        if self.user_power:
            self.btn_awaken.pack_forget()
            self.power_display.configure(text=f"Your Soul Power: {self.user_power[0]} {self.user_power[1]}", text_color=self.user_power[4])
            self.btn_cast_power.pack(pady=20)
            self.btn_cast_power.configure(fg_color=self.user_power[4])
        else:
            self.btn_cast_power.pack_forget()
            self.power_display.configure(text="Power Status: Unknown")

    def _trigger_awakening(self):
        self.btn_awaken.configure(state="disabled")
        def gacha_anim(rolls):
            if rolls > 0:
                p = random.choice(self.all_anime_powers)
                self.power_display.configure(text=f"Analyzing... {p[0]} {p[1]}", text_color="#ffffff")
                self.after(50, lambda: gacha_anim(rolls - 1))
            else:
                final_power = random.choice(self.all_anime_powers)
                self._save_profile(final_power)
                self._refresh_powerbox_ui()
        gacha_anim(30)

    def _cast_my_power(self):
        if self.user_power:
            self.start_game(self.user_power[3])

    def _build_boss_tab(self, parent):
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(expand=True, fill="both")
        
        ctk.CTkLabel(container, text="Epic Boss Battles (Real Characters)", font=ctk.CTkFont(size=36, weight="bold"), text_color="#ef4444").pack(pady=10)
        ctk.CTkLabel(container, text="Add 'madara.png', 'sukuna.png' to NithinVerse_Assets folder for HD character rendering!\n👆 Point Finger = Shoot | 👐 Open Hands = Deflect | 👊 Bring Hands Together = Smash", text_color="#9ca3af", font=ctk.CTkFont(size=14)).pack(pady=(0, 20))
        
        grid = ctk.CTkFrame(container, fg_color="transparent")
        grid.pack(expand=True)
        
        villains = [
            ("Madara Uchiha", "👁️", "Attacks with massive Fireballs.", "BOSS_MADARA", "#ef4444"),
            ("Pain", "🌀", "Drops Black Receivers.", "BOSS_PAIN", "#a855f7"),
            ("Ryomen Sukuna", "👹", "Fires invisible Cleaves.", "BOSS_SUKUNA", "#facc15"),
            ("Muzan Kibutsuji", "🧛", "Lashes out with Blood Whips.", "BOSS_MUZAN", "#f43f5e")
        ]
        
        for i, (title, icon, desc, mode, color) in enumerate(villains):
            card = ctk.CTkFrame(grid, fg_color="#111827", corner_radius=15, border_width=2, border_color=color)
            card.grid(row=0, column=i, padx=15, pady=20, ipadx=20, ipady=20, sticky="nsew")
            ctk.CTkLabel(card, text=f"{icon}\n{title}", font=ctk.CTkFont(size=24, weight="bold"), text_color=color).pack(pady=10)
            ctk.CTkLabel(card, text=desc, text_color="#9ca3af", font=ctk.CTkFont(size=13), wraplength=180).pack(pady=10)
            ctk.CTkButton(card, text="FIGHT BOSS", fg_color=color, hover_color="#0f172a", font=ctk.CTkFont(size=16, weight="bold"), height=40, command=lambda m=mode: self.start_game(m)).pack(pady=10)

    def _build_grid_tab(self, parent, data_list, header_text, header_color):
        scroll_frame = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True)
        ctk.CTkLabel(scroll_frame, text=header_text, font=ctk.CTkFont(size=24, weight="bold"), text_color=header_color).pack(pady=10)
        grid = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        grid.pack(expand=True)
        
        for i, (title, icon, desc, mode, color) in enumerate(data_list):
            card = ctk.CTkFrame(grid, fg_color="#111827", corner_radius=10, border_width=1, border_color=color)
            card.grid(row=i//5, column=i%5, padx=10, pady=10, ipadx=5, ipady=5, sticky="nsew")
            ctk.CTkLabel(card, text=f"{icon} {title}", font=ctk.CTkFont(size=14, weight="bold"), text_color=color).pack(pady=2)
            ctk.CTkLabel(card, text=desc, text_color="#9ca3af", font=ctk.CTkFont(size=10), wraplength=140).pack(pady=2)
            ctk.CTkButton(card, text="START", fg_color=color, hover_color="#0f172a", font=ctk.CTkFont(size=11, weight="bold"), height=25, command=lambda m=mode: self.start_game(m)).pack(pady=5)

    def _build_voice_tab(self, parent):
        container = ctk.CTkFrame(parent, fg_color="transparent", width=600)
        container.pack(pady=40)
        ctk.CTkLabel(container, text="🗣️ Set Your Custom Anime Dialogues", font=ctk.CTkFont(size=24, weight="bold"), text_color="#00f3ff").pack(pady=(0, 20))
        if not HAS_VOICE:
            ctk.CTkLabel(container, text="⚠️ SpeechRecognition library not found. Voice disabled.", text_color="#ef4444").pack()
            return
            
        self.entries = {}
        for action, default in self.voice_mappings.items():
            row = ctk.CTkFrame(container, fg_color="transparent")
            row.pack(fill="x", pady=5)
            ctk.CTkLabel(row, text=f"Trigger {action.upper()}:", font=ctk.CTkFont(weight="bold"), width=150, anchor="e").pack(side="left", padx=10)
            entry = ctk.CTkEntry(row, width=300)
            entry.insert(0, default)
            entry.pack(side="left", padx=10)
            self.entries[action] = entry
            
        ctk.CTkButton(container, text="💾 Save Dialogues", fg_color="#10b981", hover_color="#059669", command=self.save_voice_mappings).pack(pady=20)
        self.voice_status = ctk.CTkLabel(container, text="Mic Status: Listening...", text_color="#10b981")
        self.voice_status.pack()

    def save_voice_mappings(self):
        for action, entry in self.entries.items():
            self.voice_mappings[action] = entry.get().lower().strip()
        self.voice_status.configure(text="Settings Saved! Speak clearly into the mic.", text_color="#3b82f6")

    def _listen_for_voice(self):
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                while self.is_running:
                    try:
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                        text = self.recognizer.recognize_google(audio).lower()
                        print(f"Recognized: {text}") 
                        for action, phrase in self.voice_mappings.items():
                            if phrase in text:
                                self.active_voice_effect = action
                                self.effect_timer = 60 
                    except sr.WaitTimeoutError: pass
                    except sr.UnknownValueError: pass
                    except Exception: time.sleep(0.5) 
        except Exception as e:
            print("Microphone initialization failed. Voice commands disabled.")

    # ==========================================
    # GAME STATE MANAGEMENT & ANIMATIONS
    # ==========================================
    def go_to_menu(self):
        self.game_mode = "MENU"
        if self.vision_thread:
            self.vision_thread.set_mode(self.game_mode)
        self.btn_menu.pack_forget()
        self.slide_transition(self.game_frame, self.menu_frame)

    def start_game(self, mode):
        self.game_mode = mode
        if self.vision_thread:
            self.vision_thread.set_mode(self.game_mode)
        self.btn_menu.pack(side="right", padx=20, pady=15)
        
        self.hand_trails = [deque(maxlen=15), deque(maxlen=15)]
        self.entities = []
        self.hit_markers = []
        self.global_score = 0
        self.lives = 5
        self.combo = 0
        self.combo_timer = 0
        self.victory = False
        
        self.difficulty_multiplier = 1.0
        if "TURBO" in mode or "SPEED" in mode: self.difficulty_multiplier = 2.0
        if "SLOWMO" in mode: self.difficulty_multiplier = 0.5
        
        self.fx = {"impact_frames": 0, "show_speed_lines": False, "charge_level": 0, "domain_radius": 0, "screen_shake": 0, "capture_timer": 0, "flash_text": ""}
        
        if "BOSS" in mode:
            b_type = mode.replace("BOSS_", "")
            b_name, b_color = "The Dark Lord", (0, 0, 255)
            if b_type == "MADARA": b_name, b_color = "Madara Uchiha", (0, 0, 255)
            elif b_type == "PAIN": b_name, b_color = "Pain (Deva Path)", (255, 100, 200)
            elif b_type == "SUKUNA": b_name, b_color = "Ryomen Sukuna", (150, 150, 255)
            elif b_type == "MUZAN": b_name, b_color = "Muzan Kibutsuji", (200, 200, 200)

            self.boss = {"name": b_name, "type": b_type, "color": b_color, "hp": 3000, "max_hp": 3000, "x": 400, "y": 150, "dx": 10, "attack_timer": 30, "phase": 1}
            self.lives = 10 
            self.difficulty_multiplier = 1.5
            if not self.user_power: messagebox.showinfo("NithinVerse Power Missing", "You have not awakened your power yet! Giving you default energy.")
        elif "PONG" in mode: 
            self.ball = [{"x": 400, "y": 300, "dx": 10, "dy": 7, "radius": 15}]
            if "MULTI" in mode: 
                self.ball.append({"x": 400, "y": 200, "dx": -10, "dy": 5, "radius": 15})
                self.ball.append({"x": 400, "y": 400, "dx": 8, "dy": -8, "radius": 15})
        elif "FLAPPY" in mode: self.bird_y = 300 
        
        self.slide_transition(self.menu_frame, self.game_frame)

    def _start_camera(self):
        self.vision_thread = AsyncVisionThread(0)
        self.vision_thread.set_mode(self.game_mode)
        self.update_frame()

    # ==========================================
    # 🎭 NEW: AR COSPLAY & REAL ASSET BLENDING
    # ==========================================
    def engine_ar_cosplay(self, frame, pose_res, seg_mask, w, h):
        """Replaces background using Selfie Segmentation. Draws procedural anime outfits OR loads PNGs."""
        if seg_mask is not None:
            bg = np.zeros_like(frame)
            if "AKATSUKI" in self.game_mode:
                bg[:] = (20, 20, 20) 
                for _ in range(50): 
                    rx, ry = random.randint(0, w), random.randint(0, h)
                    cv2.line(bg, (rx, ry), (rx-5, ry+20), (100, 100, 150), 1)
            elif "TANJIRO" in self.game_mode:
                bg[:] = (30, 80, 30) 
                cv2.circle(bg, (w//2, h), 300, (20, 50, 20), -1)
            elif "GOKU" in self.game_mode:
                bg[:] = (255, 200, 100) 
                self.draw_speed_lines(bg, w, h)
            elif "GOJO" in self.game_mode:
                bg[:] = (50, 0, 0) 
                cv2.circle(bg, (w//2, h//2), int(h*0.8), (0,0,0), -1) 
            else:
                bg[:] = (150, 50, 255)

            condition = np.stack((seg_mask,) * 3, axis=-1) > 0.5
            frame = np.where(condition, frame, bg)

        if pose_res and pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks.landmark
            l_sh = (int(lm[11].x * w), int(lm[11].y * h))
            r_sh = (int(lm[12].x * w), int(lm[12].y * h))
            l_hip = (int(lm[23].x * w), int(lm[23].y * h))
            r_hip = (int(lm[24].x * w), int(lm[24].y * h))
            nose = (int(lm[0].x * w), int(lm[0].y * h))

            pts = np.array([[l_sh[0]-30, l_sh[1]-20], [r_sh[0]+30, r_sh[1]-20], [r_hip[0]+40, r_hip[1]+100], [l_hip[0]-40, l_hip[1]+100]], np.int32)
            
            # Check for Real User Assets
            skin_name = self.game_mode.replace("COSPLAY_", "")
            if skin_name in self.loaded_images:
                # User has provided a real PNG texture!
                img_t = self.loaded_images[skin_name]
                w_box = abs(r_sh[0] - l_sh[0]) + 100
                h_box = abs(r_hip[1] - l_sh[1]) + 150
                if w_box > 10 and h_box > 10:
                    frame = overlay_transparent(frame, img_t, l_sh[0]-50, l_sh[1]-50, (w_box, h_box))
            else:
                # Procedural Fallback Graphics (Vastly improved shapes)
                overlay = frame.copy()
                if "AKATSUKI" in self.game_mode:
                    cv2.fillPoly(overlay, [pts], (20, 20, 20)) 
                    cv2.circle(overlay, (l_sh[0], l_sh[1]+50), 15, (0,0,200), -1)
                    cv2.circle(overlay, (r_sh[0]-20, r_sh[1]+80), 20, (0,0,200), -1)
                    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
                elif "TANJIRO" in self.game_mode:
                    cv2.fillPoly(overlay, [pts], (50, 200, 50))
                    for i in range(5):
                        for j in range(5):
                            if (i+j)%2 == 0:
                                cx = l_sh[0] + i*40; cy = l_sh[1] + j*40
                                cv2.rectangle(overlay, (cx, cy), (cx+40, cy+40), (20, 20, 20), -1)
                    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
                elif "GOKU" in self.game_mode:
                    cv2.fillPoly(overlay, [pts], (0, 100, 255)) 
                    cv2.circle(overlay, (l_sh[0]+30, l_sh[1]+60), 30, (255,255,255), -1) 
                    cv2.circle(overlay, (w//2, h//2), 300, (0, 255, 255), 5)
                    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
                elif "GOJO" in self.game_mode:
                    cv2.fillPoly(overlay, [pts], (15, 15, 15)) 
                    cv2.line(overlay, (nose[0]-80, nose[1]-40), (nose[0]+80, nose[1]-50), (20,20,20), 40)
                    frame = cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)
                else:
                    cv2.fillPoly(overlay, [pts], (100, 100, 255))
                    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        return frame

    # ==========================================
    # CORE ENGINE UTILS
    # ==========================================
    def apply_bloom_effect(self, layer):
        h, w = layer.shape[:2]
        small = cv2.resize(layer, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST)
        blurred = cv2.blur(small, (5, 5)) 
        upscaled = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.add(layer, upscaled)

    def draw_speed_lines(self, frame, w, h):
        center = (w//2, h//2)
        for _ in range(15): 
            angle = random.uniform(0, 2 * math.pi)
            r1 = random.randint(int(h * 0.3), int(h * 0.6))
            r2 = w
            x1, y1 = int(center[0] + r1 * math.cos(angle)), int(center[1] + r1 * math.sin(angle))
            x2, y2 = int(center[0] + r2 * math.cos(angle)), int(center[1] + r2 * math.sin(angle))
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    def draw_arcade_hud(self, frame, w, h):
        cv2.putText(frame, f"SCORE: {int(self.global_score)}", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
        lives_str = "LIVES: " + ("X " * self.lives)
        cv2.putText(frame, lives_str, (20, 90), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        
        if self.combo > 1:
            pulse = int(10 * math.sin(time.time() * 15))
            cv2.putText(frame, f"{self.combo}x COMBO!", (w//2 - 120, 80 + pulse), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 165, 255), 3)
            if self.combo >= 10:
                cv2.rectangle(frame, (0,0), (w, h), (0, 165, 255), int(15 + pulse))
                self.fx["show_speed_lines"] = True
            else: self.fx["show_speed_lines"] = False
        else: self.fx["show_speed_lines"] = False

        if self.combo_timer > 0: self.combo_timer -= 1
        else: self.combo = 0 

        for hm in self.hit_markers[:]:
            scale = 0.8 + (20 - hm["life"]) * 0.05
            color = (255, 255, 255) if hm["life"] % 2 == 0 else (0, 255, 255)
            cv2.putText(frame, hm["text"], (hm["x"], hm["y"]), cv2.FONT_HERSHEY_ITALIC, scale, color, 3)
            hm["life"] -= 1; hm["y"] -= 3 
            if hm["life"] <= 0: self.hit_markers.remove(hm)

    def trigger_life_loss(self):
        self.lives -= 1
        self.combo = 0
        self.fx["screen_shake"] = 15 
        self.fx["impact_frames"] = 3 

    # ==========================================
    # MAIN UI RENDER LOOP
    # ==========================================
    def update_frame(self):
        if not self.is_running or not self.vision_thread: return
        
        try:
            ret, frame, hand_res, pose_res, seg_mask = self.vision_thread.read()
            
            if ret and frame is not None:
                h, w, _ = frame.shape
                
                if self.game_mode not in ["MENU", "GAMEOVER"] and self.lives <= 0:
                    self.game_mode = "GAMEOVER"
                    self.effect_timer = 150 
                
                if self.game_mode == "GAMEOVER":
                    overlay = np.zeros_like(frame)
                    frame = cv2.addWeighted(frame, 0.2, overlay, 0.8, 0)
                    title_text = "VICTORY!" if self.victory else "GAME OVER"
                    title_color = (0, 255, 0) if self.victory else (0, 0, 255)
                    cv2.putText(frame, title_text, (w//2 - 250, h//2 - 50), cv2.FONT_HERSHEY_DUPLEX, 3, title_color, 5)
                    cv2.putText(frame, f"FINAL SCORE: {int(self.global_score)}", (w//2 - 180, h//2 + 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)
                    self.effect_timer -= 1
                    if self.effect_timer <= 0: self.go_to_menu()
                else:
                    if any(m in self.game_mode for m in ["BREATHING", "RASENGAN", "SHIELD", "LIGHTNING", "AMATERASU", "DOMAIN", "AUTO", "AURA", "BEAM", "DISTORTION", "BOSS"]):
                        overlay = np.zeros_like(frame)
                        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
                        
                    if self.effect_timer > 0:
                        self.effect_timer -= 1
                        if self.effect_timer == 0: self.active_voice_effect = None

                    # --- ROUTING ENGINE ---
                    is_arcade = True
                    if "COSPLAY" in self.game_mode:
                        is_arcade = False
                        frame = self.engine_ar_cosplay(frame, pose_res, seg_mask, w, h)
                    elif "BOSS" in self.game_mode: frame = self.engine_boss_fight(frame, hand_res, pose_res, w, h)
                    elif "PONG" in self.game_mode: frame = self.engine_pong(frame, hand_res, w, h)
                    elif "SLASHER" in self.game_mode: frame = self.engine_slasher(frame, hand_res, w, h)
                    elif "SHURIKEN" in self.game_mode: frame = self.engine_shuriken(frame, hand_res, w, h)
                    elif "FLAPPY" in self.game_mode: frame = self.engine_flappy(frame, hand_res, w, h)
                    elif "INVADERS" in self.game_mode: frame = self.engine_invaders(frame, hand_res, w, h)
                    elif "METEOR" in self.game_mode: frame = self.engine_meteor(frame, hand_res, w, h)
                    elif "CATCHER" in self.game_mode: frame = self.engine_catcher(frame, hand_res, w, h)
                    else:
                        is_arcade = False
                        if "BREATHING" in self.game_mode: frame = self.engine_breathing(frame, hand_res, w, h)
                        elif "RASENGAN" in self.game_mode: frame = self.engine_rasengan(frame, hand_res, w, h)
                        elif "SHIELD" in self.game_mode: frame = self.engine_magic_shield(frame, hand_res, w, h)
                        elif "LIGHTNING" in self.game_mode: frame = self.engine_lightning(frame, hand_res, w, h)
                        elif "AMATERASU" in self.game_mode: frame = self.engine_amaterasu(frame, hand_res, w, h)
                        elif "DOMAIN" in self.game_mode: frame = self.engine_domain(frame, pose_res, w, h)
                        elif "BEAM" in self.game_mode: frame = self.engine_beam(frame, hand_res, pose_res, w, h)
                        elif "DISTORTION" in self.game_mode: frame = self.engine_distortion(frame, hand_res, w, h)
                        elif "AURA_HEAL" in self.game_mode: frame = self.engine_heal(frame, pose_res, w, h)
                        elif "AUTO" in self.game_mode: frame = self.engine_ai_automage(frame, hand_res, pose_res, w, h)

                    if is_arcade and self.game_mode != "MENU":
                        self.draw_arcade_hud(frame, w, h)

                    if self.fx["show_speed_lines"]:
                        self.draw_speed_lines(frame, w, h)
                        
                    if self.fx["impact_frames"] > 0:
                        frame = cv2.bitwise_not(frame)
                        self.fx["impact_frames"] -= 1
                        
                    if self.fx["screen_shake"] > 0:
                        intensity = self.fx["screen_shake"]
                        shake_x, shake_y = random.randint(-intensity, intensity), random.randint(-intensity, intensity)
                        M = np.float32([[1, 0, shake_x], [0, 1, shake_y]])
                        frame = cv2.warpAffine(frame, M, (w, h))
                        self.fx["screen_shake"] -= 2

                    if self.active_voice_effect:
                        cv2.putText(frame, f"VOICE CMD: {self.active_voice_effect.upper()}!", (w//2 - 150, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
                
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                self.video_label.imgtk = img_tk
                self.video_label.configure(image=img_tk)

        except Exception as e:
            pass # Soft catch

        self.after(20, self.update_frame) 

    # ==========================================
    # 👹 EPIC BOSS BATTLE ENGINE (REAL CHARACTERS)
    # ==========================================
    def engine_boss_fight(self, frame, hand_res, pose_res, w, h):
        glow_layer = np.zeros_like(frame)
        bx, by = int(self.boss["x"]), int(self.boss["y"])
        b_type = self.boss.get("type", "MADARA")
        pulse = int(10 * math.sin(time.time() * 10))
        
        # 1. Real Character Assets Or High-Fidelity Vectors
        if b_type in self.loaded_images:
            img_t = self.loaded_images[b_type]
            frame = overlay_transparent(frame, img_t, bx-80, by-80, (160, 160))
        else:
            if b_type == "MADARA":
                cv2.circle(glow_layer, (bx, by), 60 + pulse, (0, 0, 255), -1) 
                cv2.circle(glow_layer, (bx, by), 15, (0, 0, 0), -1) 
                for i in range(3):
                    theta = time.time()*5 + i*(2*math.pi/3)
                    tx, ty = bx + int(35*math.cos(theta)), by + int(35*math.sin(theta))
                    cv2.circle(glow_layer, (tx, ty), 10, (0, 0, 0), -1) 
                # Procedural Susanoo Ribcage
                cv2.ellipse(glow_layer, (bx, by+40), (100, 150), 0, 0, 180, (200, 0, 255), 8)
                cv2.ellipse(glow_layer, (bx, by+60), (120, 180), 0, 0, 180, (200, 0, 255), 8)
            elif b_type == "PAIN":
                cv2.circle(glow_layer, (bx, by), 60 + pulse, (200, 100, 200), -1) 
                for i in range(1, 5): cv2.circle(glow_layer, (bx, by), i*15, (0, 0, 0), 2) 
                cv2.circle(glow_layer, (bx, by), 5, (0, 0, 0), -1) 
                # Rain village headband outline
                cv2.rectangle(glow_layer, (bx-40, by-70), (bx+40, by-50), (100,100,100), -1)
            elif b_type == "SUKUNA":
                cv2.circle(glow_layer, (bx, by), 60 + pulse, (200, 220, 255), -1) 
                cv2.line(glow_layer, (bx-40, by-20), (bx-20, by), (0, 0, 0), 4) 
                cv2.line(glow_layer, (bx+40, by-20), (bx+20, by), (0, 0, 0), 4)
                cv2.line(glow_layer, (bx-40, by+20), (bx-20, by+10), (0, 0, 0), 4)
                cv2.line(glow_layer, (bx+40, by+20), (bx+20, by+10), (0, 0, 0), 4)
                cv2.circle(glow_layer, (bx-20, by-10), 6, (0, 0, 255), -1) 
                cv2.circle(glow_layer, (bx+20, by-10), 6, (0, 0, 255), -1)
                cv2.circle(glow_layer, (bx-30, by+5), 5, (0, 0, 255), -1)  
                cv2.circle(glow_layer, (bx+30, by+5), 5, (0, 0, 255), -1)
            elif b_type == "MUZAN":
                cv2.circle(glow_layer, (bx, by), 60 + pulse, (240, 240, 240), -1) 
                cv2.ellipse(glow_layer, (bx-20, by-10), (8, 4), 0, 0, 360, (0,0,255), -1)
                cv2.ellipse(glow_layer, (bx+20, by-10), (8, 4), 0, 0, 360, (0,0,255), -1)
                # Procedural Fedora Hat
                cv2.ellipse(glow_layer, (bx, by-40), (80, 20), 0, 0, 360, (20,20,20), -1)
                cv2.rectangle(glow_layer, (bx-40, by-80), (bx+40, by-40), (20,20,20), -1)
                for i in range(6):
                    theta = time.time()*3 + i*(2*math.pi/6) + math.sin(time.time()*5 + i)
                    tx, ty = bx + int(100*math.cos(theta)), by + int(100*math.sin(theta))
                    cv2.line(glow_layer, (bx, by), (tx, ty), (0, 0, 150), 6) 
            else:
                cv2.circle(glow_layer, (bx, by), 80 + pulse, (0, 0, 150), -1)
                cv2.circle(glow_layer, (bx, by), 50 + pulse, (0, 0, 255), -1)
        
        self.boss["x"] += self.boss["dx"]
        if self.boss["x"] < 100 or self.boss["x"] > w - 100: self.boss["dx"] *= -1
            
        if self.boss["hp"] < (self.boss["max_hp"] // 2) and self.boss["phase"] == 1:
            self.boss["phase"] = 2
            self.boss["dx"] = int(self.boss["dx"] * 1.5)
            self.add_hit_marker(w//2, h//2, "PHASE 2 ENRAGE!")
            self.fx["impact_frames"] = 5; self.fx["screen_shake"] = 20
            
        self.boss["attack_timer"] -= 1
        if self.boss["attack_timer"] <= 0:
            self.boss["attack_timer"] = random.randint(15, 30) if self.boss["phase"] == 2 else random.randint(25, 50)
            for _ in range(random.randint(1, self.boss["phase"])):
                self.entities.append({"type": f"boss_laser_{b_type}", "x": bx + random.randint(-50, 50), "y": by + 50})

        p_color = (255, 255, 0); p_name = "ENERGY"
        if self.user_power: p_color = self.hex_to_bgr(self.user_power[4]); p_name = self.user_power[0]

        action = None
        hx, hy = -1, -1
        if hand_res and hand_res.multi_hand_landmarks:
            hands = hand_res.multi_hand_landmarks
            states = [self.get_finger_state(h_lm) for h_lm in hands]
            
            if len(hands) == 2:
                s1, s2 = states[0], states[1]
                dist = math.hypot(hands[0].landmark[0].x - hands[1].landmark[0].x, hands[0].landmark[0].y - hands[1].landmark[0].y) * w
                cx, cy = int((hands[0].landmark[0].x + hands[1].landmark[0].x)/2 * w), int((hands[0].landmark[0].y + hands[1].landmark[0].y)/2 * h)
                hx, hy = cx, cy
                
                if sum(s1[1:]) >= 3 and sum(s2[1:]) >= 3 and dist > 250: action = "SHIELD"
                elif dist < 150: action = "CHARGE"
            elif len(hands) == 1:
                s1 = states[0]
                hx, hy = int(hands[0].landmark[8].x * w), int(hands[0].landmark[8].y * h)
                if s1[1] == 1 and sum(s1[2:]) == 0: action = "SHOOT"
        
        if action == "SHIELD":
            self.create_intricate_mandala(glow_layer, hx, hy, 200, time.time()*3, p_color)
            for e in self.entities:
                if e["type"].startswith("boss_laser_") and math.hypot(e["x"] - hx, e["y"] - hy) < 150:
                    e["type"] = "deflected"; e["dy"] = -30 
        elif action == "CHARGE":
            cv2.circle(glow_layer, (hx, hy), 80, p_color, -1)
            cv2.circle(glow_layer, (hx, hy), 60, (255, 255, 255), -1)
            if math.hypot(hx - bx, hy - by) < 160:
                self.boss["hp"] -= 10; self.global_score += 10
                if random.random() < 0.2: self.add_hit_marker(bx, by, f"{p_name} SMASH!"); self.fx["screen_shake"] = 3
        elif action == "SHOOT":
            cv2.line(glow_layer, (hx, hy), (hx, 0), p_color, 15)
            cv2.line(glow_layer, (hx, hy), (hx, 0), (255, 255, 255), 5)
            if abs(hx - bx) < 80:
                self.boss["hp"] -= 3; self.global_score += 5
                if random.random() < 0.2: self.add_hit_marker(bx + random.randint(-50,50), by + random.randint(-50,50), "HIT!")

        for e in self.entities[:]:
            if e["type"].startswith("boss_laser_"):
                e["y"] += 15 * self.difficulty_multiplier
                ptype = e["type"].replace("boss_laser_", "")
                
                if ptype == "MADARA":
                    cv2.circle(glow_layer, (int(e["x"]), int(e["y"])), 25, (0, 100, 255), -1) 
                    cv2.circle(glow_layer, (int(e["x"]), int(e["y"])), 12, (0, 255, 255), -1)
                elif ptype == "PAIN":
                    cv2.line(glow_layer, (int(e["x"]), int(e["y"]-20)), (int(e["x"]), int(e["y"]+20)), (0, 0, 0), 8) 
                    cv2.circle(glow_layer, (int(e["x"]), int(e["y"])), 5, (255, 0, 255), -1)
                elif ptype == "SUKUNA":
                    cv2.ellipse(glow_layer, (int(e["x"]), int(e["y"])), (30, 10), random.randint(0,180), 0, 180, (255, 255, 255), 4) 
                elif ptype == "MUZAN":
                    cv2.line(glow_layer, (int(e["x"]), int(e["y"]-30)), (int(e["x"]+random.randint(-15,15)), int(e["y"]+30)), (0, 0, 150), 6) 
                else: cv2.circle(glow_layer, (int(e["x"]), int(e["y"])), 15, (0, 0, 255), -1) 
                
                if e["y"] > h:
                    if e in self.entities: self.entities.remove(e)
                    self.trigger_life_loss()
            elif e["type"] == "deflected":
                e["y"] += e.get("dy", -15)
                cv2.circle(glow_layer, (int(e["x"]), int(e["y"])), 15, p_color, -1)
                if math.hypot(e["x"] - bx, e["y"] - by) < 80:
                    self.boss["hp"] -= 20; self.global_score += 50
                    self.add_hit_marker(bx, by, "DEFLECT DAMAGE!")
                    if e in self.entities: self.entities.remove(e)
                elif e["y"] < 0:
                    if e in self.entities: self.entities.remove(e)

        cv2.putText(frame, self.boss["name"], (w//2 - 120, 35), cv2.FONT_HERSHEY_DUPLEX, 1, self.boss["color"], 2)
        cv2.rectangle(frame, (w//2 - 300, 50), (w//2 + 300, 70), (0, 0, 0), -1)
        hp_ratio = max(0, self.boss["hp"] / self.boss["max_hp"])
        cv2.rectangle(frame, (w//2 - 300, 50), (w//2 - 300 + int(600 * hp_ratio), 70), self.boss["color"], -1)
        
        frame = cv2.add(frame, self.apply_bloom_effect(glow_layer))
        
        if self.boss["hp"] <= 0:
            self.global_score += 10000
            self.victory = True
            self.game_mode = "GAMEOVER"
            self.effect_timer = 150
            self.fx["impact_frames"] = 5
        
        return frame

    # ==========================================
    # 🧠 THE "AI FREE THINKER" (AUTO-MAGE)
    # ==========================================
    def get_finger_state(self, hand_lm):
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]
        state = []
        if hand_lm.landmark[tips[0]].x > hand_lm.landmark[pips[0]].x: state.append(1)
        else: state.append(0)
        for i in range(1, 5):
            state.append(1 if hand_lm.landmark[tips[i]].y < hand_lm.landmark[pips[i]].y else 0)
        return state

    def engine_ai_automage(self, frame, hand_res, pose_res, w, h):
        action = None
        cv2.putText(frame, "AI SENSORS ONLINE: AWAITING GESTURE", (w//2 - 250, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
        
        if hand_res and hand_res.multi_hand_landmarks:
            hands = hand_res.multi_hand_landmarks
            states = [self.get_finger_state(h_lm) for h_lm in hands]
            
            if len(hands) == 2:
                s1, s2 = states[0], states[1]
                dist = math.hypot(hands[0].landmark[0].x - hands[1].landmark[0].x, hands[0].landmark[0].y - hands[1].landmark[0].y) * w
                if sum(s1[1:]) >= 3 and sum(s2[1:]) >= 3 and dist > 250: action = "SHIELD"
                elif dist < 150: action = "RASENGAN"
                elif sum(s1[1:]) == 0 and sum(s2[1:]) == 0: action = "DOMAIN"
            elif len(hands) == 1:
                s1 = states[0]
                if s1[1] == 1 and sum(s1[2:]) == 0: action = "AMATERASU"
                elif s1[1] == 1 and s1[4] == 1 and s1[2] == 0 and s1[3] == 0: action = "LIGHTNING"
                elif sum(s1[1:]) >= 3: action = "BREATHING_FIRE" 

        if action == "SHIELD": return self.engine_magic_shield(frame, hand_res, w, h)
        elif action == "RASENGAN": return self.engine_rasengan(frame, hand_res, w, h)
        elif action == "AMATERASU": return self.engine_amaterasu(frame, hand_res, w, h)
        elif action == "LIGHTNING": return self.engine_lightning(frame, hand_res, w, h)
        elif action == "BREATHING": return self.engine_breathing(frame, hand_res, w, h)
        elif action == "DOMAIN": return self.engine_domain(frame, pose_res, w, h)
        return frame

    # ==========================================
    # 🌌 HYPER-REALISTIC PARAMETERIZED ANIME VFX
    # ==========================================
    def get_color_from_mode(self, default=(0, 165, 255)):
        if "FIRE" in self.game_mode or "SUN" in self.game_mode: return (0, 0, 255) 
        elif "WATER" in self.game_mode or "AZURE" in self.game_mode: return (255, 150, 0) 
        elif "THUNDER" in self.game_mode or "LIGHT" in self.game_mode: return (0, 255, 255) 
        elif "WIND" in self.game_mode or "TAO" in self.game_mode: return (100, 255, 100) 
        elif "STONE" in self.game_mode or "MAGNET" in self.game_mode: return (150, 150, 150) 
        elif "SERPENT" in self.game_mode: return (50, 200, 50) 
        elif "FLOWER" in self.game_mode or "EMBODIMENT" in self.game_mode: return (200, 100, 255) 
        elif "INSECT" in self.game_mode or "GALICK" in self.game_mode: return (255, 0, 150) 
        elif "SOUND" in self.game_mode or "CRIMSON" in self.game_mode: return (100, 0, 255) 
        elif "VOID" in self.game_mode or "CERO" in self.game_mode: return (50, 50, 50) 
        return default

    def engine_distortion(self, frame, hand_res, w, h):
        active = False
        if hand_res and hand_res.multi_hand_landmarks:
            tip = hand_res.multi_hand_landmarks[0].landmark[8]
            px, py = int(tip.x * w), int(tip.y * h)
            active = True
            
            radius = 150
            if py > radius and py < h-radius and px > radius and px < w-radius:
                roi = frame[py-radius:py+radius, px-radius:px+radius]
                map_x, map_y = np.meshgrid(np.arange(radius*2), np.arange(radius*2))
                map_x = map_x - radius; map_y = map_y - radius
                r, theta = cv2.cartToPolar(map_x, map_y)
                theta += r / 20.0 * math.sin(time.time() * 5) 
                map_x, map_y = cv2.polarToCart(r, theta)
                map_x = np.clip(map_x + radius, 0, radius*2-1).astype(np.float32)
                map_y = np.clip(map_y + radius, 0, radius*2-1).astype(np.float32)
                swirled = cv2.remap(roi, map_x, map_y, interpolation=cv2.INTER_LINEAR)
                frame[py-radius:py+radius, px-radius:px+radius] = swirled
                cv2.circle(frame, (px, py), 15, (0,0,0), -1)

        if active: cv2.putText(frame, "KAMUI!", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (50, 50, 50), 2)
        return frame

    def engine_lightning(self, frame, hand_res, w, h):
        glow_layer = np.zeros_like(frame)
        active = False
        if self.active_voice_effect == "lightning": active = True
        
        c1 = self.get_color_from_mode((255, 200, 100))
        c2 = self.get_color_from_mode((255, 100, 0))
        
        if hand_res and hand_res.multi_hand_landmarks:
            for hand_lm in hand_res.multi_hand_landmarks:
                px, py = int(hand_lm.landmark[0].x * w), int(hand_lm.landmark[0].y * h)
                active = True
                for _ in range(4):
                    pts = [(px, py)]
                    curr_x, curr_y = px, py
                    for _ in range(5):
                        curr_x += random.randint(-80, 80)
                        curr_y += random.randint(-80, 80)
                        pts.append((curr_x, curr_y))
                    pts = np.array(pts, np.int32)
                    cv2.polylines(glow_layer, [pts], False, c1, 5) 
                    cv2.polylines(glow_layer, [pts], False, c2, 12)

        if active:
            frame = cv2.add(frame, self.apply_bloom_effect(glow_layer))
            self.fx["screen_shake"] = 5
            cv2.putText(frame, "CHIDORI!", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, c2, 2)
        return frame

    def engine_amaterasu(self, frame, hand_res, w, h):
        glow_layer = np.zeros_like(frame)
        active = False
        if self.active_voice_effect == "black_flame": active = True
        
        if hand_res and hand_res.multi_hand_landmarks:
            tip = hand_res.multi_hand_landmarks[0].landmark[8]
            px, py = int(tip.x * w), int(tip.y * h)
            active = True
            if random.random() < 0.5:
                self.entities.append({"x": px, "y": py - 30, "size": random.randint(20, 50), "life": 100})

        for f in self.entities[:]:
            f["y"] -= 3
            f["x"] += random.randint(-5, 5)
            f["life"] -= 5
            if f["life"] > 0:
                cv2.circle(glow_layer, (f["x"], f["y"]), f["size"], (150, 0, 150), -1) 
                cv2.circle(frame, (f["x"], f["y"]), int(f["size"] * 0.6), (0, 0, 0), -1) 
            else:
                self.entities.remove(f)

        if active or len(self.entities) > 0:
            frame = cv2.add(frame, self.apply_bloom_effect(glow_layer))
            cv2.putText(frame, "AMATERASU!", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 3)
        return frame

    def engine_domain(self, frame, pose_res, w, h):
        active = False
        if self.active_voice_effect == "shield": active = True
        color = self.get_color_from_mode((255, 255, 255))
        
        if pose_res and pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks.landmark
            l_wrist = (int(lm[15].x * w), int(lm[15].y * h))
            r_wrist = (int(lm[16].x * w), int(lm[16].y * h))
            if abs(l_wrist[0] - r_wrist[0]) < 100 and abs(l_wrist[1] - r_wrist[1]) < 100:
                active = True

        if active:
            if self.fx["domain_radius"] == 0: self.fx["impact_frames"] = 2
            self.fx["domain_radius"] = min(self.fx["domain_radius"] + 40, w) 
            
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (w//2, h//2), self.fx["domain_radius"], 255, -1)
            
            inverted = cv2.bitwise_not(frame)
            frame = np.where(mask[:, :, None] == 255, inverted, frame)
            
            cv2.circle(frame, (w//2, h//2), self.fx["domain_radius"], color, 5)
            cv2.putText(frame, "DOMAIN EXPANSION!", (w//2 - 150, 50), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        else:
            self.fx["domain_radius"] = 0
        return frame

    def engine_breathing(self, frame, hand_res, w, h):
        glow_layer = np.zeros_like(frame) 
        is_active = False
        if self.active_voice_effect == "flame": is_active = True
        color = self.get_color_from_mode((0, 165, 255))
        
        if hand_res and hand_res.multi_hand_landmarks:
            tip = hand_res.multi_hand_landmarks[0].landmark[8]
            px, py = int(tip.x * w), int(tip.y * h)
            self.hand_trails[0].appendleft((px, py))
            is_active = True
        
        if is_active and len(self.hand_trails[0]) > 2:
            trail = self.hand_trails[0]
            for j in range(1, len(trail)):
                size = int(35 * (1 - j/len(trail)))
                if size > 0:
                    pt1, pt2 = trail[j-1], trail[j]
                    cv2.line(glow_layer, pt1, pt2, (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6)), size + 10)
                    cv2.line(glow_layer, pt1, pt2, color, size)
                    cv2.line(glow_layer, pt1, pt2, (255, 255, 255), size // 2)
                    
                    if random.random() < 0.3:
                        sx, sy = pt1[0] + random.randint(-40, 40), pt1[1] + random.randint(-40, 40)
                        cv2.circle(glow_layer, (sx, sy), random.randint(2, 6), color, -1)

            frame = cv2.add(frame, self.apply_bloom_effect(glow_layer))
            cv2.putText(frame, "BREATHING FORM!", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        return frame

    def engine_rasengan(self, frame, hand_res, w, h):
        glow_layer = np.zeros_like(frame)
        active = False
        cx, cy = w//2, h//2
        if self.active_voice_effect == "rasengan": active = True
        color = self.get_color_from_mode((255, 200, 50))
        
        if hand_res and hand_res.multi_hand_landmarks and len(hand_res.multi_hand_landmarks) == 2:
            lm1 = hand_res.multi_hand_landmarks[0].landmark[0] 
            lm2 = hand_res.multi_hand_landmarks[1].landmark[0] 
            dist = math.hypot(lm1.x - lm2.x, lm1.y - lm2.y) * w
            if dist < 250: 
                active = True
                cx, cy = int((lm1.x + lm2.x)/2 * w), int((lm1.y + lm2.y)/2 * h)

        if active:
            if self.fx["charge_level"] == 0: self.fx["impact_frames"] = 3
            self.fx["charge_level"] += 1
            
            radius = 70 + int(5 * math.sin(time.time() * 20))
            if "GIANT" in self.game_mode or "PLANET" in self.game_mode: radius += 50
            
            cv2.circle(glow_layer, (cx, cy), radius + 20, (int(color[0]*0.5), int(color[1]*0.5), int(color[2]*0.5)), -1)
            cv2.circle(glow_layer, (cx, cy), radius, color, -1)
            cv2.circle(glow_layer, (cx, cy), radius - 30, (255, 255, 255), -1)
            
            for i in range(5):
                angle = time.time() * 15 + (i * 1.25)
                cv2.ellipse(glow_layer, (cx, cy), (radius+10, radius-20), math.degrees(angle), 0, 180, (255, 255, 255), 4)
                cv2.ellipse(glow_layer, (cx, cy), (radius-20, radius+10), math.degrees(-angle), 0, 180, color, 4)

            frame = cv2.add(frame, self.apply_bloom_effect(glow_layer))
            self.fx["screen_shake"] = 3
            cv2.putText(frame, "CHAKRA SPHERE!", (cx-100, cy+130), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
        else:
            self.fx["charge_level"] = 0
        return frame

    def engine_beam(self, frame, hand_res, pose_res, w, h):
        firing = False
        cy, cx = h // 2, w // 2
        color = self.get_color_from_mode((255, 255, 0)) 
        
        if self.active_voice_effect == "beam": firing = True
        
        if hand_res and hand_res.multi_hand_landmarks and len(hand_res.multi_hand_landmarks) == 2:
            lm1 = hand_res.multi_hand_landmarks[0].landmark
            lm2 = hand_res.multi_hand_landmarks[1].landmark
            if lm1[12].y < lm1[0].y and lm2[12].y < lm2[0].y: 
                dist = math.hypot(lm1[0].x - lm2[0].x, lm1[0].y - lm2[0].y) * w
                if dist < 200: 
                    firing = True
                    cx = int((lm1[0].x + lm2[0].x)/2 * w)
                    cy = int((lm1[0].y + lm2[0].y)/2 * h)
        
        if firing:
            if self.fx["charge_level"] == 0: self.fx["impact_frames"] = 4 
            self.fx["charge_level"] += 1
            self.fx["show_speed_lines"] = True
            
            beam_layer = frame.copy()
            width = 250
            if "X10" in self.game_mode or "FINAL" in self.game_mode: width = 400
            
            pts_outer = np.array([[cx-80, cy], [cx+80, cy], [cx+width, 0], [cx-width, 0]], np.int32)
            cv2.polylines(beam_layer, [pts_outer], True, color, 20)
            
            pts_core = np.array([[cx-50, cy], [cx+50, cy], [cx+(width-50), 0], [cx-(width-50), 0]], np.int32)
            cv2.fillPoly(beam_layer, [pts_core], (255, 255, 255)) 
            
            radius = random.randint(60, 90)
            cv2.circle(beam_layer, (cx, cy), radius, (255, 255, 255), -1)
            cv2.circle(beam_layer, (cx, cy), radius+15, color, 6)
            
            for r in range(1, 4):
                angle = time.time() * 200 * r
                cv2.ellipse(beam_layer, (cx, cy), (radius + r*25, 30 + r*15), angle, 0, 360, color, 4)
            
            pulse = 0.6 + 0.2 * math.sin(time.time() * 40)
            frame = cv2.addWeighted(beam_layer, pulse, frame, 1 - pulse, 0)
            
            shake_x, shake_y = random.randint(-12, 12), random.randint(-12, 12)
            M = np.float32([[1, 0, shake_x], [0, 1, shake_y]])
            frame = cv2.warpAffine(frame, M, (w, h))
            cv2.putText(frame, "ENERGY BEAM!", (cx-120, cy+120), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
        else:
            self.fx["charge_level"] = 0
            self.fx["show_speed_lines"] = False
            
        return frame

    def create_intricate_mandala(self, canvas, cx, cy, size, angle, color):
        r = size // 2
        cv2.circle(canvas, (cx, cy), r, color, 4)
        cv2.circle(canvas, (cx, cy), r - 15, color, 2)
        cv2.circle(canvas, (cx, cy), r - 60, color, 2)
        
        for i in range(3):
            theta = angle + (i * math.pi/3)
            pts = []
            for j in range(4):
                px = cx + int((r-15) * math.cos(theta + j*math.pi/2))
                py = cy + int((r-15) * math.sin(theta + j*math.pi/2))
                pts.append([px, py])
            cv2.polylines(canvas, [np.array(pts, np.int32)], True, color, 2)
            
        star_pts = []
        for i in range(6):
            theta = -angle * 1.5 + (i * math.pi/3)
            px = cx + int((r-60) * math.cos(theta))
            py = cy + int((r-60) * math.sin(theta))
            star_pts.append([px, py])
        cv2.fillPoly(canvas, [np.array(star_pts, np.int32)], (int(color[0]*0.5), int(color[1]*0.5), int(color[2]*0.5)))
        cv2.polylines(canvas, [np.array(star_pts, np.int32)], True, color, 3)

    def engine_magic_shield(self, frame, hand_res, w, h):
        glow_layer = np.zeros_like(frame) 
        shield_active = False
        if self.active_voice_effect == "shield": shield_active = True
        color = self.get_color_from_mode((0, 165, 255))
        
        if hand_res and hand_res.multi_hand_landmarks:
            for hand_lm in hand_res.multi_hand_landmarks:
                if hand_lm.landmark[8].y < hand_lm.landmark[6].y and hand_lm.landmark[12].y < hand_lm.landmark[10].y:
                    shield_active = True
                    px, py = int(hand_lm.landmark[9].x * w), int(hand_lm.landmark[9].y * h)
                    
                    size = 300
                    shield_square = np.zeros((size, size, 3), dtype=np.uint8)
                    self.create_intricate_mandala(shield_square, size//2, size//2, size, time.time()*2, color)
                    
                    M = cv2.getRotationMatrix2D((size//2, size//2), 0, 1)
                    M[1, 1] = 0.65 
                    tilted_shield = cv2.warpAffine(shield_square, M, (size, size))
                    
                    y1, y2 = max(0, py - size//2), min(h, py + size//2)
                    x1, x2 = max(0, px - size//2), min(w, px + size//2)
                    
                    sy1, sy2 = size//2 - (py - y1), size//2 + (y2 - py)
                    sx1, sx2 = size//2 - (px - x1), size//2 + (x2 - px)
                    
                    if y2 > y1 and x2 > x1:
                        roi = glow_layer[y1:y2, x1:x2]
                        shield_roi = tilted_shield[sy1:sy2, sx1:sx2]
                        glow_layer[y1:y2, x1:x2] = cv2.add(roi, shield_roi) 

        if shield_active:
            frame = cv2.add(frame, self.apply_bloom_effect(glow_layer))
            cv2.putText(frame, "ABSOLUTE DEFENSE!", (w//2 - 150, h - 50), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        return frame

    # ==========================================
    # 🎮 VISION GAMES (50+ Configs) - Arcade Logic
    # ==========================================
    def engine_catcher(self, frame, results, w, h):
        self.difficulty_multiplier += 0.0005 
        if random.random() < 0.03 * self.difficulty_multiplier:
            t = 1 if random.random() < 0.7 else -1
            self.entities.append({"type": "fall", "val": t, "x": random.randint(50, w-50), "y": 0, "dy": random.randint(5, 10)})
            
        hx, hy, is_open = -1, -1, False
        if results and results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            hx, hy = int(lm[9].x * w), int(lm[9].y * h)
            is_open = lm[8].y < lm[6].y and lm[12].y < lm[10].y
            cv2.circle(frame, (hx, hy), 40, (0, 255, 0) if is_open else (0,0,255), 2)

        for e in self.entities[:]:
            e["y"] += int(e["dy"] * self.difficulty_multiplier)
            color = (0, 255, 255) if e["val"] == 1 else (0, 0, 255)
            cv2.circle(frame, (e["x"], e["y"]), 20, color, -1)
            
            if is_open and hx > 0 and math.hypot(hx - e["x"], hy - e["y"]) < 50:
                if e["val"] == 1:
                    self.combo += 1; self.combo_timer = 60; self.global_score += 10 * self.combo
                    self.add_hit_marker(e["x"], e["y"], "CAUGHT!")
                else: self.trigger_life_loss()
                self.entities.remove(e)
            elif e["y"] > h:
                if e["val"] == 1: self.trigger_life_loss() 
                self.entities.remove(e)
        return frame

    def engine_shuriken(self, frame, results, w, h):
        self.difficulty_multiplier += 0.0005 
        color = self.get_color_from_mode((200, 200, 200))
        
        targets = [e for e in self.entities if e.get("type") == "target"]
        
        if random.random() < 0.02 * self.difficulty_multiplier and len(targets) < 4:
            self.entities.append({"type": "target", "x": random.randint(100, w-100), "y": 50, "dx": random.choice([-5, 5]) * self.difficulty_multiplier})
            
        hx, hy = -1, -1
        if results and results.multi_hand_landmarks:
            hx = int(results.multi_hand_landmarks[0].landmark[0].x * w)
            hy = int(results.multi_hand_landmarks[0].landmark[0].y * h)
            self.hand_trails[0].appendleft(hy)
            
            if len(self.hand_trails[0]) > 5:
                vel = self.hand_trails[0][4] - self.hand_trails[0][0]
                if vel > 80: 
                    self.entities.append({"type": "shuriken", "x": hx, "y": hy, "angle": 0})
                    self.hand_trails[0].clear() 

        if hx > 0: cv2.circle(frame, (hx, hy), 20, (0, 255, 0), 2)

        for e in self.entities[:]:
            if e["type"] == "target":
                e["x"] += int(e["dx"])
                if e["x"] < 50 or e["x"] > w-50: e["dx"] *= -1
                cv2.rectangle(frame, (e["x"]-30, e["y"]-30), (e["x"]+30, e["y"]+30), (0, 0, 255), -1)
                
            elif e["type"] == "shuriken":
                e["y"] -= 25 
                e["angle"] += 30 
                pts = []
                for i in range(4):
                    theta = math.radians(e["angle"] + i*90)
                    pts.append([e["x"] + int(20*math.cos(theta)), e["y"] + int(20*math.sin(theta))])
                cv2.fillPoly(frame, [np.array(pts, np.int32)], color)
                
                for t in targets:
                    if math.hypot(e["x"] - t["x"], e["y"] - t["y"]) < 40:
                        self.combo += 1; self.combo_timer = 60; self.global_score += 50 * self.combo
                        self.add_hit_marker(t["x"], t["y"], random.choice(["BAM!", "CRITICAL!", "SHARP!"]))
                        if t in self.entities: self.entities.remove(t)
                        if e in self.entities: self.entities.remove(e)
                        break
                        
                if e in self.entities and e["y"] < 0: 
                    self.entities.remove(e); self.trigger_life_loss()
        return frame

    def engine_pong(self, frame, results, w, h):
        self.difficulty_multiplier += 0.0005
        color = self.get_color_from_mode((255, 255, 0))
        
        py = h // 2 
        if results and results.multi_hand_landmarks:
            py = int(results.multi_hand_landmarks[0].landmark[8].y * h)
            cv2.circle(frame, (int(results.multi_hand_landmarks[0].landmark[8].x * w), py), 15, (0, 255, 100), -1)

        player_pad = [w - 40, py - 60, w - 20, py + 60]
        
        for b in self.ball:
            b["x"] += int(b["dx"] * self.difficulty_multiplier)
            b["y"] += int(b["dy"] * self.difficulty_multiplier)
            if b["y"] <= 15 or b["y"] >= h - 15: b["dy"] *= -1
            
            ai_speed = 8 * self.difficulty_multiplier
            if self.ai_paddle_y < self.ball[0]["y"] - 20: self.ai_paddle_y += ai_speed
            elif self.ai_paddle_y > self.ball[0]["y"] + 20: self.ai_paddle_y -= ai_speed
            ai_pad = [20, int(self.ai_paddle_y) - 60, 40, int(self.ai_paddle_y) + 60]

            bx, by = b["x"], b["y"]
            if bx + 15 >= player_pad[0] and player_pad[1] < by < player_pad[3]:
                b["dx"] *= -1.1; b["x"] = player_pad[0] - 20
                self.combo += 1; self.combo_timer = 90; self.global_score += 10 * self.combo
                self.add_hit_marker(int(bx), int(by), "RALLY!")
                
            if bx - 15 <= ai_pad[2] and ai_pad[1] < by < ai_pad[3]:
                b["dx"] *= -1; b["x"] = ai_pad[2] + 20

            if bx < 0: 
                self.global_score += 100 * self.combo
                b["x"] = w//2; b["y"] = h//2; b["dx"] = -10
                self.add_hit_marker(w//2, h//2, "GOAL!")
            elif bx > w: 
                self.trigger_life_loss()
                b["x"] = w//2; b["y"] = h//2; b["dx"] = 10

            cv2.circle(frame, (int(bx), int(by)), 15, color, -1)
            
        cv2.line(frame, (w//2, 0), (w//2, h), (255, 255, 255), 2)
        cv2.rectangle(frame, (player_pad[0], player_pad[1]), (player_pad[2], player_pad[3]), (0, 255, 100), -1) 
        cv2.rectangle(frame, (ai_pad[0], ai_pad[1]), (ai_pad[2], ai_pad[3]), (0, 100, 255), -1) 
        return frame

    def engine_slasher(self, frame, results, w, h):
        self.difficulty_multiplier += 0.001
        color = self.get_color_from_mode((255, 0, 255))
        
        if random.random() < 0.05 * self.difficulty_multiplier:
            speed = random.randint(5, 12) * self.difficulty_multiplier
            t = 1
            if "BOMB" in self.game_mode and random.random() < 0.3: t = -1
            
            y_start = 0
            if "REVERSE" in self.game_mode: 
                y_start = h; speed = -speed
            self.entities.append({"x": random.randint(100, w-100), "y": y_start, "dy": speed, "active": True, "type": t})
            
        hx, hy = -1, -1
        if results and results.multi_hand_landmarks:
            hx = int(results.multi_hand_landmarks[0].landmark[8].x * w)
            hy = int(results.multi_hand_landmarks[0].landmark[8].y * h)
            self.hand_trails[0].appendleft((hx, hy))
            for j in range(1, len(self.hand_trails[0])):
                cv2.line(frame, self.hand_trails[0][j-1], self.hand_trails[0][j], color, 15 - j)

        for f in self.entities[:]:
            f["y"] += int(f["dy"])
            if f["active"]:
                c = (0, 255, 0) if f["type"] == 1 else (0, 0, 255)
                cv2.circle(frame, (f["x"], f["y"]), 30, c, -1)
                
                if hx > 0 and math.hypot(hx - f["x"], hy - f["y"]) < 40:
                    f["active"] = False
                    if f["type"] == 1:
                        self.combo += 1; self.combo_timer = 60; self.global_score += 10 * self.combo
                        self.add_hit_marker(f["x"], f["y"], random.choice(["SLASH!", "CUT!", "COMBO!"]))
                        self.fx["impact_frames"] = 1
                    else: self.trigger_life_loss() 
            else:
                cv2.circle(frame, (f["x"]-20, f["y"]), 15, (0, 200, 0), -1)
                cv2.circle(frame, (f["x"]+20, f["y"]), 15, (0, 200, 0), -1)
                
            if f["y"] > h or f["y"] < -50: 
                if f["active"] and f["type"] == 1: self.trigger_life_loss()
                self.entities.remove(f)

        return frame

    def engine_flappy(self, frame, results, w, h):
        self.difficulty_multiplier += 0.0005
        pipe_speed = int(8 * self.difficulty_multiplier)
        color = self.get_color_from_mode((0, 255, 255))
        
        if len(self.entities) == 0 or self.entities[-1]["x"] < w - 300:
            gap_y = random.randint(200, h-200)
            self.entities.append({"x": w, "gap_y": gap_y, "passed": False})
            
        if results and results.multi_hand_landmarks:
            self.ball[0]["y"] = int(results.multi_hand_landmarks[0].landmark[8].y * h)
        cv2.circle(frame, (self.ball[0]["x"], self.ball[0]["y"]), 20, color, -1)
        
        for p in self.entities[:]:
            p["x"] -= pipe_speed
            cv2.rectangle(frame, (p["x"], 0), (p["x"]+50, p["gap_y"]-70), (0, 255, 0), -1)
            cv2.rectangle(frame, (p["x"], p["gap_y"]+70), (p["x"]+50, h), (0, 255, 0), -1)
            
            if p["x"] < self.ball[0]["x"] + 20 and p["x"] + 50 > self.ball[0]["x"] - 20:
                if self.ball[0]["y"] < p["gap_y"] - 70 or self.ball[0]["y"] > p["gap_y"] + 70: 
                    self.trigger_life_loss(); p["x"] -= 200 
            elif not p["passed"] and p["x"] < self.ball[0]["x"]:
                p["passed"] = True; self.combo += 1; self.combo_timer = 90
                self.global_score += 20 * self.combo
                self.add_hit_marker(p["x"], self.ball[0]["y"], "PASS!")
                
            if p["x"] < -50: self.entities.remove(p)

        return frame

    def engine_invaders(self, frame, results, w, h):
        self.difficulty_multiplier += 0.001
        color = self.get_color_from_mode((0, 255, 255))
        
        targets = [e for e in self.entities if e.get("type") == "enemy"]
        
        if random.random() < 0.03 * self.difficulty_multiplier:
            self.entities.append({"type": "enemy", "x": random.randint(50, w-50), "y": 0})
            
        px = w//2
        if results and results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            px = int(lm[9].x * w)
            fingers_open = lm[8].y < lm[6].y or lm[12].y < lm[10].y
            if not fingers_open and len([e for e in self.entities if e["type"]=="laser"]) < 3:
                self.entities.append({"type": "laser", "x": px, "y": h-80})

        cv2.rectangle(frame, (px-30, h-60), (px+30, h-20), (255, 0, 0), -1)
        for e in self.entities[:]:
            if e["type"] == "enemy":
                e["y"] += int(4 * self.difficulty_multiplier)
                cv2.rectangle(frame, (e["x"]-20, e["y"]-20), (e["x"]+20, e["y"]+20), (0, 0, 255), -1)
                if e["y"] > h: 
                    self.entities.remove(e); self.trigger_life_loss() 
            elif e["type"] == "laser":
                e["y"] -= 15
                cv2.line(frame, (e["x"], e["y"]), (e["x"], e["y"]+20), color, 5)
                if e["y"] < 0: self.entities.remove(e)
                else:
                    for en in targets:
                        if math.hypot(e["x"] - en["x"], e["y"] - en["y"]) < 30:
                            if en in self.entities: self.entities.remove(en)
                            if e in self.entities: self.entities.remove(e)
                            self.combo += 1; self.combo_timer = 60; self.global_score += 15 * self.combo
                            self.add_hit_marker(e["x"], e["y"], "BOOM!")
                            break

        return frame

    def engine_meteor(self, frame, results, w, h):
        self.difficulty_multiplier += 0.0005
        color = self.get_color_from_mode((0, 100, 255))
        
        if len(self.entities) < int(3 * self.difficulty_multiplier):
            self.entities.append({"x": random.randint(100, w-100), "y": random.randint(100, h-100), "life": 100})
            
        hands = []
        if results and results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                hands.append((int(lm.landmark[9].x * w), int(lm.landmark[9].y * h)))
                cv2.circle(frame, hands[-1], 30, (255, 100, 0), 3)

        for m in self.entities[:]:
            m["life"] -= int(1 * self.difficulty_multiplier) 
            cv2.circle(frame, (m["x"], m["y"]), int(m["life"]/2), color, -1)
            
            for hx, hy in hands:
                if math.hypot(hx - m["x"], hy - m["y"]) < int(m["life"]/2) + 20:
                    self.combo += 1; self.combo_timer = 50; self.global_score += 25 * self.combo
                    self.add_hit_marker(m["x"], m["y"], "SMASH!")
                    self.fx["impact_frames"] = 2
                    self.entities.remove(m); break
                    
            if m in self.entities and m["life"] <= 0:
                self.trigger_life_loss(); self.entities.remove(m)

        return frame

    def _reset_ball(self, w, h, direction):
        self.ball[0]["x"] = w // 2; self.ball[0]["y"] = h // 2
        self.ball[0]["dx"] = 10 * direction; self.ball[0]["dy"] = random.choice([-7, 7])

    def on_closing(self):
        self.is_running = False
        if self.vision_thread: self.vision_thread.release()
        self.destroy()

if __name__ == "__main__":
    try:
        app = NithinGamersApp()
        app.protocol("WM_DELETE_WINDOW", app.on_closing)
        app.mainloop()
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        error_msg = f"Application Crash:\n{str(e)}\n\n{traceback.format_exc()}"
        messagebox.showerror("Fatal Error", error_msg)
        sys.exit(1)