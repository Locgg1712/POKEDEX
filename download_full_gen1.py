from icrawler.builtin import BingImageCrawler
import os
import time

pokemon_list = [
"bulbasaur","ivysaur","venusaur",
"charmander","charmeleon","charizard",
"squirtle","wartortle","blastoise",
"caterpie","butterfree",
"weedle","beedrill",
"pidgey","pidgeot",
"rattata","raticate",
"spearow","fearow",
"ekans","arbok",
"pikachu","raichu",
"sandshrew","sandslash",
"nidoran-f","nidoqueen",
"nidoran-m","nidoking",
"clefairy","clefable",
"vulpix","ninetales",
"jigglypuff","wigglytuff",
"zubat","golbat",
"oddish","gloom","vileplume",
"paras","parasect",
"diglett","dugtrio",
"meowth","persian",
"psyduck","golduck",
"mankey","primeape"
]

for name in pokemon_list:
    print(f"🔥 Đang tải: {name}")
    
    save_dir = f"data/{name}"
    os.makedirs(save_dir, exist_ok=True)
    
    crawler = BingImageCrawler(
        downloader_threads=4,
        storage={'root_dir': save_dir}
    )
    
    crawler.crawl(
        keyword=f"{name} pokemon official artwork",  
        max_num=50,
        min_size=(200, 200)  # 👈 lọc ảnh nhỏ
    )
    
    time.sleep(1)

print("✅ DONE DATASET")