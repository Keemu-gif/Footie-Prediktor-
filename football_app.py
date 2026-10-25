"""
Football Predictor - desktop app
Open this file in IDLE (comes with Python) and press F5 to run it.
The first run installs the extra pieces it needs automatically.
"""

import subprocess
import sys


# ---------------------------------------------------------------- auto-install
def ensure_packages():
    needed = []
    for pkg in ("numpy", "pandas", "scipy"):
        try:
            __import__(pkg)
        except ImportError:
            needed.append(pkg)
    if needed:
        print("Installing", ", ".join(needed), "- this only happens once, please wait...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *needed])


ensure_packages()

import io
import threading
import urllib.request
from datetime import date
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

BASE = "https://www.football-data.co.uk"
HALF_LIFE_DAYS = 180
MAX_GOALS = 10
LEAGUES = {
    "England - Premier League": "E0",
    "England - Championship": "E1",
    "Spain - La Liga": "SP1",
    "Germany - Bundesliga": "D1",
    "Italy - Serie A": "I1",
    "France - Ligue 1": "F1",
    "Netherlands - Eredivisie": "N1",
    "Portugal - Primeira Liga": "P1",
    "Scotland - Premiership": "SC0",
    "Belgium - Pro League": "B1",
    "Turkey - Super Lig": "T1",
    "Greece - Super League": "G1",
}


# ---------------------------------------------------------------- data
def fetch_csv(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            text = r.read().decode("latin-1")
        df = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
        df.columns = [c.strip().lstrip("﻿") for c in df.columns]
        return df
    except Exception:
        return None


def season_codes(n=3):
    today = date.today()
    start = today.year if today.month >= 7 else today.year - 1
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(start - n + 1, start + 1)]


def load_results(league):
    frames = [f for c in season_codes() if (f := fetch_csv(f"{BASE}/mmz4281/{c}/{league}.csv")) is not None]
    if not frames:
        raise RuntimeError("Could not download results. Check your internet connection.")
    df = pd.concat(frames, ignore_index=True)
    df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].dropna()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df[["FTHG", "FTAG"]] = df[["FTHG", "FTAG"]].astype(int)
    return df


def load_fixtures(league):
    df = fetch_csv(f"{BASE}/fixtures.csv")
    if df is None or "Div" not in df.columns:
        return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam"])
    return df[df["Div"] == league][["Date", "HomeTeam", "AwayTeam"]]


# ---------------------------------------------------------------- model
class PoissonModel:
    def fit(self, df):
        ref = df["Date"].max()
        self.teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        h, a = df.HomeTeam.map(idx).values, df.AwayTeam.map(idx).values
        hg, ag = df.FTHG.values, df.FTAG.values
        w = 0.5 ** ((ref - df.Date).dt.days.values / HALF_LIFE_DAYS)

        def nll(p):
            att = p[:n] - p[:n].mean()
            dfn, home = p[n:2 * n], p[-1]
            lam = np.exp(att[h] + dfn[a] + home)
            mu = np.exp(att[a] + dfn[h])
            ll = hg * np.log(lam) - lam + ag * np.log(mu) - mu
            return -np.sum(w * ll) + 0.01 * np.sum(p[:-1] ** 2)

        x0 = np.zeros(2 * n + 1)
        x0[-1] = 0.25
        p = minimize(nll, x0, method="L-BFGS-B").x
        self.att = dict(zip(self.teams, p[:n] - p[:n].mean()))
        self.dfn = dict(zip(self.teams, p[n:2 * n]))
        self.home = p[-1]
        return self

    def predict(self, home, away):
        lam = np.exp(self.att[home] + self.dfn[away] + self.home)
        mu = np.exp(self.att[away] + self.dfn[home])
        g = np.arange(MAX_GOALS + 1)
        grid = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        hs, as_ = np.unravel_index(grid.argmax(), grid.shape)
        hw, dr, aw = np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()
        pick = ["Home win", "Draw", "Away win"][int(np.argmax([hw, dr, aw]))]
        over = 1 - sum(grid[i, j] for i in g for j in g if i + j <= 2)
        btts = 1 - grid[0, :].sum() - grid[:, 0].sum() + grid[0, 0]
        return dict(score=f"{hs}-{as_}", home=hw, draw=dr, away=aw, pick=pick,
                    xg=f"{lam:.1f} - {mu:.1f}", over=over, btts=btts)


# ---------------------------------------------------------------- app window
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Football Predictor")
        self.geometry("1000x620")
        self.model = None

        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="League:").pack(side="left")
        self.league = ttk.Combobox(top, values=list(LEAGUES), state="readonly", width=30)
        self.league.current(0)
        self.league.pack(side="left", padx=6)
        self.go = ttk.Button(top, text="Load & predict", command=self.start)
        self.go.pack(side="left")
        self.status = ttk.Label(top, text="Choose a league and click 'Load & predict'.")
        self.status.pack(side="left", padx=12)

        cols = ("date", "home", "away", "score", "h", "d", "a", "pick", "xg", "over", "btts")
        heads = ("Date", "Home", "Away", "Likely score", "Home %", "Draw %", "Away %",
                 "Prediction", "Exp. goals", "Over 2.5", "Both score")
        widths = (80, 130, 130, 85, 65, 65, 65, 90, 80, 65, 75)
        self.table = ttk.Treeview(self, columns=cols, show="headings", height=16)
        for c, hd, wd in zip(cols, heads, widths):
            self.table.heading(c, text=hd)
            self.table.column(c, width=wd, anchor="center")
        self.table.pack(fill="both", expand=True, padx=10)

        bottom = ttk.LabelFrame(self, text="Predict any match", padding=10)
        bottom.pack(fill="x", padx=10, pady=10)
        self.home_box = ttk.Combobox(bottom, state="readonly", width=22)
        self.away_box = ttk.Combobox(bottom, state="readonly", width=22)
        ttk.Label(bottom, text="Home:").pack(side="left")
        self.home_box.pack(side="left", padx=4)
        ttk.Label(bottom, text="Away:").pack(side="left")
        self.away_box.pack(side="left", padx=4)
        ttk.Button(bottom, text="Predict", command=self.predict_one).pack(side="left", padx=6)
        self.one = ttk.Label(bottom, text="", font=("TkDefaultFont", 10, "bold"))
        self.one.pack(side="left", padx=10)

    def add_row(self, when, home, away, p):
        self.table.insert("", "end", values=(
            when, home, away, p["score"], f"{p['home']:.0%}", f"{p['draw']:.0%}",
            f"{p['away']:.0%}", p["pick"], p["xg"], f"{p['over']:.0%}", f"{p['btts']:.0%}"))

    def start(self):
        self.go.config(state="disabled")
        self.status.config(text="Downloading results and training... (about 10-30 seconds)")
        self.table.delete(*self.table.get_children())
        threading.Thread(target=self.work, args=(LEAGUES[self.league.get()],), daemon=True).start()

    def work(self, code):
        try:
            df = load_results(code)
            model = PoissonModel().fit(df)
            fixtures = load_fixtures(code)
            self.after(0, self.done, model, fixtures, len(df))
        except Exception as e:
            self.after(0, self.failed, str(e))

    def done(self, model, fixtures, n):
        self.model = model
        self.home_box.config(values=model.teams)
        self.away_box.config(values=model.teams)
        shown = 0
        for _, f in fixtures.iterrows():
            if f.HomeTeam in model.att and f.AwayTeam in model.att:
                self.add_row(f.Date, f.HomeTeam, f.AwayTeam, model.predict(f.HomeTeam, f.AwayTeam))
                shown += 1
        msg = f"Trained on {n} past matches. {shown} upcoming fixtures predicted."
        if shown == 0:
            msg += " No fixtures listed yet - use 'Predict any match' below."
        self.status.config(text=msg)
        self.go.config(state="normal")

    def failed(self, err):
        self.status.config(text="Something went wrong.")
        self.go.config(state="normal")
        messagebox.showerror("Error", err)

    def predict_one(self):
        h, a = self.home_box.get(), self.away_box.get()
        if not self.model:
            messagebox.showinfo("First step", "Click 'Load & predict' first.")
            return
        if not h or not a or h == a:
            messagebox.showinfo("Pick teams", "Choose two different teams.")
            return
        p = self.model.predict(h, a)
        self.one.config(text=f"{p['pick']} - likely {p['score']}   "
                             f"(Home {p['home']:.0%} / Draw {p['draw']:.0%} / Away {p['away']:.0%})")
        self.add_row("custom", h, a, p)


if __name__ == "__main__":
    App().mainloop()
