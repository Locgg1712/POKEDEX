# src/evolution.py
"""
Hệ thống phân cấp họ tiến hóa (Evolution Families & Hierarchical Logic)
cho 150 loài Pokémon Thế hệ 1.

Cung cấp:
1. EVOLUTION_FAMILIES: Ánh xạ 78 họ tiến hóa với thứ tự các giai đoạn (Stage 1 -> Stage 2 -> Stage 3).
2. POKEMON_TO_FAMILY: Tra cứu nhanh Pokémon thuộc họ nào và ở stage mấy.
3. BODY_ARCHETYPES: Phân nhóm theo hình thái học (Quadruped, Bipedal, Avian, Serpentine, Aquatic, Insectoid, Spherical).
4. evolution_rerank: Thuật toán tái xếp hạng xác suất dựa trên tổng hợp bằng chứng cùng dòng họ.
5. extract_stage_features: Các đặc trưng DSP tập trung phân biệt các bậc tiến hóa trong cùng một họ.
"""

import numpy as np
import cv2

# ==============================================================================
# 1. BẢNG TIẾN HÓA CHI TIẾT 150 POKÉMON THẾ HỆ 1 (78 HỌ TIẾN HÓA)
# ==============================================================================

EVOLUTION_FAMILIES = {
    # --- Starter Pokémon (3 stages) ---
    "bulbasaur_family": {
        "name": "Dòng Bulbasaur",
        "members": ["Bulbasaur", "Ivysaur", "Venusaur"],
        "stages": {"Bulbasaur": 1, "Ivysaur": 2, "Venusaur": 3},
        "archetype": "Quadruped",
        "types": ["Grass", "Poison"]
    },
    "charmander_family": {
        "name": "Dòng Charmander",
        "members": ["Charmander", "Charmeleon", "Charizard"],
        "stages": {"Charmander": 1, "Charmeleon": 2, "Charizard": 3},
        "archetype": "Bipedal",
        "types": ["Fire", "Flying"]
    },
    "squirtle_family": {
        "name": "Dòng Squirtle",
        "members": ["Squirtle", "Wartortle", "Blastoise"],
        "stages": {"Squirtle": 1, "Wartortle": 2, "Blastoise": 3},
        "archetype": "Bipedal",
        "types": ["Water"]
    },

    # --- Early Bug & Bird (3 stages) ---
    "caterpie_family": {
        "name": "Dòng Caterpie",
        "members": ["Caterpie", "Metapod", "Butterfree"],
        "stages": {"Caterpie": 1, "Metapod": 2, "Butterfree": 3},
        "archetype": "Insectoid",
        "types": ["Bug", "Flying"]
    },
    "weedle_family": {
        "name": "Dòng Weedle",
        "members": ["Weedle", "Kakuna", "Beedrill"],
        "stages": {"Weedle": 1, "Kakuna": 2, "Beedrill": 3},
        "archetype": "Insectoid",
        "types": ["Bug", "Poison"]
    },
    "pidgey_family": {
        "name": "Dòng Pidgey",
        "members": ["Pidgey", "Pidgeotto", "Pidgeot"],
        "stages": {"Pidgey": 1, "Pidgeotto": 2, "Pidgeot": 3},
        "archetype": "Avian",
        "types": ["Normal", "Flying"]
    },

    # --- 2 stages Common Lines ---
    "rattata_family": {
        "name": "Dòng Rattata",
        "members": ["Rattata", "Raticate"],
        "stages": {"Rattata": 1, "Raticate": 2},
        "archetype": "Quadruped",
        "types": ["Normal"]
    },
    "spearow_family": {
        "name": "Dòng Spearow",
        "members": ["Spearow", "Fearow"],
        "stages": {"Spearow": 1, "Fearow": 2},
        "archetype": "Avian",
        "types": ["Normal", "Flying"]
    },
    "ekans_family": {
        "name": "Dòng Ekans",
        "members": ["Ekans", "Arbok"],
        "stages": {"Ekans": 1, "Arbok": 2},
        "archetype": "Serpentine",
        "types": ["Poison"]
    },
    "pikachu_family": {
        "name": "Dòng Pikachu",
        "members": ["Pikachu", "Raichu"],
        "stages": {"Pikachu": 1, "Raichu": 2},
        "archetype": "Bipedal",
        "types": ["Electric"]
    },
    "sandshrew_family": {
        "name": "Dòng Sandshrew",
        "members": ["Sandshrew", "Sandslash", "Alolan Sandslash"],
        "stages": {"Sandshrew": 1, "Sandslash": 2, "Alolan Sandslash": 2},
        "archetype": "Bipedal",
        "types": ["Ground", "Ice"]
    },
    "nidoran_f_family": {
        "name": "Dòng Nidoran Nữ",
        "members": ["Nidorina", "Nidoqueen"],
        "stages": {"Nidorina": 2, "Nidoqueen": 3},
        "archetype": "Bipedal",
        "types": ["Poison", "Ground"]
    },
    "nidoran_m_family": {
        "name": "Dòng Nidoran Nam",
        "members": ["Nidorino", "Nidoking"],
        "stages": {"Nidorino": 2, "Nidoking": 3},
        "archetype": "Bipedal",
        "types": ["Poison", "Ground"]
    },
    "clefairy_family": {
        "name": "Dòng Clefairy",
        "members": ["Clefairy", "Clefable"],
        "stages": {"Clefairy": 1, "Clefable": 2},
        "archetype": "Bipedal",
        "types": ["Fairy"]
    },
    "vulpix_family": {
        "name": "Dòng Vulpix",
        "members": ["Vulpix", "Ninetales"],
        "stages": {"Vulpix": 1, "Ninetales": 2},
        "archetype": "Quadruped",
        "types": ["Fire"]
    },
    "jigglypuff_family": {
        "name": "Dòng Jigglypuff",
        "members": ["Jigglypuff", "Wigglytuff"],
        "stages": {"Jigglypuff": 1, "Wigglytuff": 2},
        "archetype": "Spherical",
        "types": ["Normal", "Fairy"]
    },
    "zubat_family": {
        "name": "Dòng Zubat",
        "members": ["Zubat", "Golbat"],
        "stages": {"Zubat": 1, "Golbat": 2},
        "archetype": "Avian",
        "types": ["Poison", "Flying"]
    },
    "oddish_family": {
        "name": "Dòng Oddish",
        "members": ["Oddish", "Gloom", "Vileplume"],
        "stages": {"Oddish": 1, "Gloom": 2, "Vileplume": 3},
        "archetype": "Bipedal",
        "types": ["Grass", "Poison"]
    },
    "paras_family": {
        "name": "Dòng Paras",
        "members": ["Paras", "Parasect"],
        "stages": {"Paras": 1, "Parasect": 2},
        "archetype": "Insectoid",
        "types": ["Bug", "Grass"]
    },
    "venonat_family": {
        "name": "Dòng Venonat",
        "members": ["Venonat", "Venomoth"],
        "stages": {"Venonat": 1, "Venomoth": 2},
        "archetype": "Insectoid",
        "types": ["Bug", "Poison"]
    },
    "diglett_family": {
        "name": "Dòng Diglett",
        "members": ["Diglett", "Dugtrio"],
        "stages": {"Diglett": 1, "Dugtrio": 2},
        "archetype": "Spherical",
        "types": ["Ground"]
    },
    "meowth_family": {
        "name": "Dòng Meowth",
        "members": ["Meowth", "Persian"],
        "stages": {"Meowth": 1, "Persian": 2},
        "archetype": "Quadruped",
        "types": ["Normal"]
    },
    "psyduck_family": {
        "name": "Dòng Psyduck",
        "members": ["Psyduck", "Golduck"],
        "stages": {"Psyduck": 1, "Golduck": 2},
        "archetype": "Bipedal",
        "types": ["Water"]
    },
    "mankey_family": {
        "name": "Dòng Mankey",
        "members": ["Mankey", "Primeape"],
        "stages": {"Mankey": 1, "Primeape": 2},
        "archetype": "Bipedal",
        "types": ["Fighting"]
    },
    "growlithe_family": {
        "name": "Dòng Growlithe",
        "members": ["Growlithe", "Arcanine"],
        "stages": {"Growlithe": 1, "Arcanine": 2},
        "archetype": "Quadruped",
        "types": ["Fire"]
    },
    "poliwag_family": {
        "name": "Dòng Poliwag",
        "members": ["Poliwag", "Poliwhirl", "Poliwrath"],
        "stages": {"Poliwag": 1, "Poliwhirl": 2, "Poliwrath": 3},
        "archetype": "Bipedal",
        "types": ["Water", "Fighting"]
    },
    "abra_family": {
        "name": "Dòng Abra",
        "members": ["Abra", "Kadabra", "Alakazam"],
        "stages": {"Abra": 1, "Kadabra": 2, "Alakazam": 3},
        "archetype": "Bipedal",
        "types": ["Psychic"]
    },
    "machop_family": {
        "name": "Dòng Machop",
        "members": ["Machop", "Machoke", "Machamp"],
        "stages": {"Machop": 1, "Machoke": 2, "Machamp": 3},
        "archetype": "Bipedal",
        "types": ["Fighting"]
    },
    "bellsprout_family": {
        "name": "Dòng Bellsprout",
        "members": ["Bellsprout", "Weepinbell", "Victreebel"],
        "stages": {"Bellsprout": 1, "Weepinbell": 2, "Victreebel": 3},
        "archetype": "Bipedal",
        "types": ["Grass", "Poison"]
    },
    "tentacool_family": {
        "name": "Dòng Tentacool",
        "members": ["Tentacool", "Tentacruel"],
        "stages": {"Tentacool": 1, "Tentacruel": 2},
        "archetype": "Aquatic",
        "types": ["Water", "Poison"]
    },
    "geodude_family": {
        "name": "Dòng Geodude",
        "members": ["Geodude", "Graveler", "Golem"],
        "stages": {"Geodude": 1, "Graveler": 2, "Golem": 3},
        "archetype": "Spherical",
        "types": ["Rock", "Ground"]
    },
    "ponyta_family": {
        "name": "Dòng Ponyta",
        "members": ["Ponyta", "Rapidash"],
        "stages": {"Ponyta": 1, "Rapidash": 2},
        "archetype": "Quadruped",
        "types": ["Fire"]
    },
    "slowpoke_family": {
        "name": "Dòng Slowpoke",
        "members": ["Slowpoke", "Slowbro"],
        "stages": {"Slowpoke": 1, "Slowbro": 2},
        "archetype": "Quadruped",
        "types": ["Water", "Psychic"]
    },
    "magnemite_family": {
        "name": "Dòng Magnemite",
        "members": ["Magnemite", "Magneton"],
        "stages": {"Magnemite": 1, "Magneton": 2},
        "archetype": "Spherical",
        "types": ["Electric", "Steel"]
    },
    "doduo_family": {
        "name": "Dòng Doduo",
        "members": ["Doduo", "Dodrio"],
        "stages": {"Doduo": 1, "Dodrio": 2},
        "archetype": "Avian",
        "types": ["Normal", "Flying"]
    },
    "seel_family": {
        "name": "Dòng Seel",
        "members": ["Seel", "Dewgong"],
        "stages": {"Seel": 1, "Dewgong": 2},
        "archetype": "Aquatic",
        "types": ["Water", "Ice"]
    },
    "grimer_family": {
        "name": "Dòng Grimer",
        "members": ["Grimer", "Muk"],
        "stages": {"Grimer": 1, "Muk": 2},
        "archetype": "Spherical",
        "types": ["Poison"]
    },
    "shellder_family": {
        "name": "Dòng Shellder",
        "members": ["Shellder", "Cloyster"],
        "stages": {"Shellder": 1, "Cloyster": 2},
        "archetype": "Aquatic",
        "types": ["Water", "Ice"]
    },
    "gastly_family": {
        "name": "Dòng Gastly",
        "members": ["Gastly", "Haunter", "Gengar"],
        "stages": {"Gastly": 1, "Haunter": 2, "Gengar": 3},
        "archetype": "Spherical",
        "types": ["Ghost", "Poison"]
    },
    "drowzee_family": {
        "name": "Dòng Drowzee",
        "members": ["Drowzee", "Hypno"],
        "stages": {"Drowzee": 1, "Hypno": 2},
        "archetype": "Bipedal",
        "types": ["Psychic"]
    },
    "krabby_family": {
        "name": "Dòng Krabby",
        "members": ["Krabby", "Kingler"],
        "stages": {"Krabby": 1, "Kingler": 2},
        "archetype": "Aquatic",
        "types": ["Water"]
    },
    "voltorb_family": {
        "name": "Dòng Voltorb",
        "members": ["Voltorb", "Electrode"],
        "stages": {"Voltorb": 1, "Electrode": 2},
        "archetype": "Spherical",
        "types": ["Electric"]
    },
    "exeggcute_family": {
        "name": "Dòng Exeggcute",
        "members": ["Exeggcute", "Exeggutor"],
        "stages": {"Exeggcute": 1, "Exeggutor": 2},
        "archetype": "Bipedal",
        "types": ["Grass", "Psychic"]
    },
    "cubone_family": {
        "name": "Dòng Cubone",
        "members": ["Cubone", "Marowak"],
        "stages": {"Cubone": 1, "Marowak": 2},
        "archetype": "Bipedal",
        "types": ["Ground"]
    },
    "hitmon_family": {
        "name": "Dòng Hitmon",
        "members": ["Hitmonlee", "Hitmonchan"],
        "stages": {"Hitmonlee": 1, "Hitmonchan": 1},
        "archetype": "Bipedal",
        "types": ["Fighting"]
    },
    "koffing_family": {
        "name": "Dòng Koffing",
        "members": ["Koffing", "Weezing"],
        "stages": {"Koffing": 1, "Weezing": 2},
        "archetype": "Spherical",
        "types": ["Poison"]
    },
    "rhyhorn_family": {
        "name": "Dòng Rhyhorn",
        "members": ["Rhyhorn", "Rhydon"],
        "stages": {"Rhyhorn": 1, "Rhydon": 2},
        "archetype": "Quadruped",
        "types": ["Ground", "Rock"]
    },
    "horsea_family": {
        "name": "Dòng Horsea",
        "members": ["Horsea", "Seadra"],
        "stages": {"Horsea": 1, "Seadra": 2},
        "archetype": "Aquatic",
        "types": ["Water"]
    },
    "goldeen_family": {
        "name": "Dòng Goldeen",
        "members": ["Goldeen", "Seaking"],
        "stages": {"Goldeen": 1, "Seaking": 2},
        "archetype": "Aquatic",
        "types": ["Water"]
    },
    "staryu_family": {
        "name": "Dòng Staryu",
        "members": ["Staryu", "Starmie"],
        "stages": {"Staryu": 1, "Starmie": 2},
        "archetype": "Aquatic",
        "types": ["Water", "Psychic"]
    },
    "magikarp_family": {
        "name": "Dòng Magikarp",
        "members": ["Magikarp", "Gyarados"],
        "stages": {"Magikarp": 1, "Gyarados": 2},
        "archetype": "Aquatic",
        "types": ["Water", "Flying"]
    },
    "eevee_family": {
        "name": "Dòng Eevee (Phân nhánh)",
        "members": ["Eevee", "Vaporeon", "Jolteon", "Flareon"],
        "stages": {"Eevee": 1, "Vaporeon": 2, "Jolteon": 2, "Flareon": 2},
        "archetype": "Quadruped",
        "types": ["Normal", "Water", "Electric", "Fire"]
    },
    "omanyte_family": {
        "name": "Dòng Omanyte",
        "members": ["Omanyte", "Omastar"],
        "stages": {"Omanyte": 1, "Omastar": 2},
        "archetype": "Aquatic",
        "types": ["Rock", "Water"]
    },
    "kabuto_family": {
        "name": "Dòng Kabuto",
        "members": ["Kabuto", "Kabutops"],
        "stages": {"Kabuto": 1, "Kabutops": 2},
        "archetype": "Bipedal",
        "types": ["Rock", "Water"]
    },
    "dratini_family": {
        "name": "Dòng Dratini",
        "members": ["Dratini", "Dragonair", "Dragonite"],
        "stages": {"Dratini": 1, "Dragonair": 2, "Dragonite": 3},
        "archetype": "Serpentine",
        "types": ["Dragon", "Flying"]
    },

    # --- Standalone / Legendary / Single Evolution Families ---
    "farfetchd_family": {"name": "Farfetchd", "members": ["Farfetchd"], "stages": {"Farfetchd": 1}, "archetype": "Avian", "types": ["Normal", "Flying"]},
    "lickitung_family": {"name": "Lickitung", "members": ["Lickitung"], "stages": {"Lickitung": 1}, "archetype": "Bipedal", "types": ["Normal"]},
    "chansey_family":   {"name": "Chansey", "members": ["Chansey"], "stages": {"Chansey": 1}, "archetype": "Bipedal", "types": ["Normal"]},
    "kangaskhan_family": {"name": "Kangaskhan", "members": ["Kangaskhan"], "stages": {"Kangaskhan": 1}, "archetype": "Bipedal", "types": ["Normal"]},
    "mr_mime_family":   {"name": "MrMime", "members": ["MrMime"], "stages": {"MrMime": 1}, "archetype": "Bipedal", "types": ["Psychic", "Fairy"]},
    "scyther_family":   {"name": "Scyther", "members": ["Scyther"], "stages": {"Scyther": 1}, "archetype": "Insectoid", "types": ["Bug", "Flying"]},
    "jynx_family":      {"name": "Jynx", "members": ["Jynx"], "stages": {"Jynx": 1}, "archetype": "Bipedal", "types": ["Ice", "Psychic"]},
    "electabuzz_family": {"name": "Electabuzz", "members": ["Electabuzz"], "stages": {"Electabuzz": 1}, "archetype": "Bipedal", "types": ["Electric"]},
    "magmar_family":    {"name": "Magmar", "members": ["Magmar"], "stages": {"Magmar": 1}, "archetype": "Bipedal", "types": ["Fire"]},
    "pinsir_family":    {"name": "Pinsir", "members": ["Pinsir"], "stages": {"Pinsir": 1}, "archetype": "Insectoid", "types": ["Bug"]},
    "tauros_family":    {"name": "Tauros", "members": ["Tauros"], "stages": {"Tauros": 1}, "archetype": "Quadruped", "types": ["Normal"]},
    "lapras_family":    {"name": "Lapras", "members": ["Lapras"], "stages": {"Lapras": 1}, "archetype": "Aquatic", "types": ["Water", "Ice"]},
    "ditto_family":     {"name": "Ditto", "members": ["Ditto"], "stages": {"Ditto": 1}, "archetype": "Spherical", "types": ["Normal"]},
    "porygon_family":   {"name": "Porygon", "members": ["Porygon"], "stages": {"Porygon": 1}, "archetype": "Bipedal", "types": ["Normal"]},
    "aerodactyl_family": {"name": "Aerodactyl", "members": ["Aerodactyl"], "stages": {"Aerodactyl": 1}, "archetype": "Avian", "types": ["Rock", "Flying"]},
    "snorlax_family":   {"name": "Snorlax", "members": ["Snorlax"], "stages": {"Snorlax": 1}, "archetype": "Bipedal", "types": ["Normal"]},
    "onix_family":      {"name": "Onix", "members": ["Onix"], "stages": {"Onix": 1}, "archetype": "Serpentine", "types": ["Rock", "Ground"]},
    "tangela_family":   {"name": "Tangela", "members": ["Tangela"], "stages": {"Tangela": 1}, "archetype": "Spherical", "types": ["Grass"]},
    "articuno_family":  {"name": "Articuno", "members": ["Articuno"], "stages": {"Articuno": 1}, "archetype": "Avian", "types": ["Ice", "Flying"]},
    "zapdos_family":    {"name": "Zapdos", "members": ["Zapdos"], "stages": {"Zapdos": 1}, "archetype": "Avian", "types": ["Electric", "Flying"]},
    "moltres_family":   {"name": "Moltres", "members": ["Moltres"], "stages": {"Moltres": 1}, "archetype": "Avian", "types": ["Fire", "Flying"]},
    "mewtwo_family":    {"name": "Mewtwo", "members": ["Mewtwo"], "stages": {"Mewtwo": 1}, "archetype": "Bipedal", "types": ["Psychic"]},
    "mew_family":       {"name": "Mew", "members": ["Mew"], "stages": {"Mew": 1}, "archetype": "Bipedal", "types": ["Psychic"]},
}

# ==============================================================================
# 2. XÂY DỰNG INDEX TRA CỨU NHANH
# ==============================================================================

POKEMON_TO_FAMILY = {}
POKEMON_STAGE_INFO = {}

for fam_id, data in EVOLUTION_FAMILIES.items():
    for member in data["members"]:
        # Chuẩn hóa tên (cả dạng thường và nguyên bản)
        key = member.lower().replace(" ", "").replace("-", "")
        POKEMON_TO_FAMILY[member] = fam_id
        POKEMON_TO_FAMILY[key] = fam_id
        stage = data["stages"].get(member, 1)
        POKEMON_STAGE_INFO[member] = {
            "family_id": fam_id,
            "family_name": data["name"],
            "stage": stage,
            "total_stages": len(data["members"]),
            "archetype": data["archetype"],
            "types": data["types"],
            "siblings": [m for m in data["members"] if m != member],
        }


def get_pokemon_family(name: str):
    """Lấy thông tin họ tiến hóa từ tên Pokémon."""
    if name in POKEMON_TO_FAMILY:
        return POKEMON_TO_FAMILY[name]
    key = name.lower().replace(" ", "").replace("-", "")
    return POKEMON_TO_FAMILY.get(key, "unknown_family")


def get_family_info(name: str):
    """Lấy toàn bộ metadata dòng họ của Pokémon."""
    if name in POKEMON_STAGE_INFO:
        return POKEMON_STAGE_INFO[name]
    key = name.lower().replace(" ", "").replace("-", "")
    for m, info in POKEMON_STAGE_INFO.items():
        if m.lower().replace(" ", "").replace("-", "") == key:
            return info
    return None


def get_all_families_map():
    """Trả về dictionary {family_id: [members]}."""
    return {fam_id: data["members"] for fam_id, data in EVOLUTION_FAMILIES.items()}


# ==============================================================================
# 3. ĐẶC TRƯNG DSP PHÂN BIỆT STAGE TIẾN HÓA (INTRA-FAMILY DSP FEATURES)
# ==============================================================================

def extract_evolution_stage_features(img):
    """
    Trích xuất đặc trưng hình học & biên chi tiết dùng để phân biệt các bậc tiến hóa:
    - Stage 1 thường nhỏ, tròn, ít góc nhọn, đường biên mịn.
    - Stage 2/3 phát triển thêm cánh, sừng, răng nanh, đuôi dài, diện tích lớn hơn,
      độ phức tạp đường biên (isoperimetric quotient) tăng cao.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    total_area = float(h * w)

    # 1. Canny edge density & complexity
    edges = cv2.Canny(gray, 50, 150)
    edge_count = float(np.count_nonzero(edges))
    edge_density = edge_count / total_area

    # 2. Contour analysis
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        perimeter = float(cv2.arcLength(c, True))
        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull))

        area_ratio = area / total_area
        hull_ratio = area / max(hull_area, 1.0)
        # Độ gồ ghề / độ phức tạp biên: P^2 / (4*pi*A) (hình tròn = 1, nhiều gai/cánh > 1)
        circularity = (perimeter * perimeter) / (4.0 * np.pi * max(area, 1.0))

        # Bounding box aspect ratio
        bx, by, bw, bh = cv2.boundingRect(c)
        aspect_ratio = float(bw) / max(float(bh), 1.0)
        extent = area / max(float(bw * bh), 1.0)
    else:
        area_ratio = 0.0
        hull_ratio = 1.0
        circularity = 1.0
        aspect_ratio = 1.0
        extent = 0.0

    # 3. Sắc thái màu: Đếm số vùng màu chính (evolution cao thường thêm dải màu mới)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    hist_h, _ = np.histogram(h_channel, bins=18, range=(0, 180))
    hist_h_norm = hist_h.astype(np.float32) / (np.sum(hist_h) + 1e-8)
    entropy_color = -float(np.sum(hist_h_norm * np.log2(hist_h_norm + 1e-12)))

    features = np.array([
        area_ratio,
        edge_density,
        hull_ratio,
        min(circularity, 50.0) / 10.0,
        aspect_ratio,
        extent,
        entropy_color / 4.0
    ], dtype=np.float32)

    return np.nan_to_num(features)


# ==============================================================================
# 4. EVOLUTION-AWARE RERANKING
# ==============================================================================

def evolution_rerank(probs, labels_dict, alpha=0.35, uncertainty_threshold=0.55):
    """
    Tái sắp xếp phân phối xác suất dự đoán dựa trên bằng chứng họ tiến hóa:
    1. Gom xác suất của tất cả thành viên trong cùng họ.
    2. Nếu tổng xác suất họ (family_evidence) vượt trội nhưng các thành viên bị chia nhỏ điểm
       (ví dụ: Pidgey 30%, Pidgeotto 28%, Pidgeot 12% -> tổng họ 70%),
       thuật toán sẽ củng cố họ này và điều chỉnh phân phối theo thành viên dẫn đầu.
    3. Giữ nguyên nếu model dự đoán tự tin với điểm cao (> uncertainty_threshold).
    """
    reranked = np.array(probs, dtype=np.float64, copy=True)
    top1_idx = int(np.argmax(reranked))
    top1_conf = float(reranked[top1_idx])

    # Nếu Top 1 đã rất chắc chắn thì không can thiệp nhiều
    if top1_conf >= uncertainty_threshold:
        return reranked / np.sum(reranked)

    # Nhóm các index theo family
    family_groups = {}
    for idx, name in labels_dict.items():
        fam_id = get_pokemon_family(name)
        if fam_id not in family_groups:
            family_groups[fam_id] = []
        family_groups[fam_id].append(idx)

    # Tính tổng xác suất cho mỗi family
    family_mass = {}
    for fam_id, member_indices in family_groups.items():
        family_mass[fam_id] = sum(reranked[i] for i in member_indices)

    # Tìm family có khối lượng xác suất lớn nhất
    best_fam = max(family_mass, key=family_mass.get)
    best_mass = family_mass[best_fam]

    # Nếu family này có tổng bằng chứng đáng kể (> 0.35)
    if best_mass > 0.35 and len(family_groups[best_fam]) > 1:
        members = family_groups[best_fam]
        # Boost nhẹ thành viên có xác suất cao nhất trong family đó
        best_member_idx = max(members, key=lambda i: reranked[i])
        boost = alpha * (best_mass / len(members))
        reranked[best_member_idx] += boost

        # Đồng thời boost tỷ lệ thuận cho các anh em trong họ
        for m_idx in members:
            if m_idx != best_member_idx:
                reranked[m_idx] += boost * 0.4

    # Chuẩn hóa lại tổng xác suất = 1
    total = np.sum(reranked)
    if total > 0:
        reranked /= total

    return reranked
