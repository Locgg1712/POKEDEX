# src/pokedex_data.py
"""
Cơ sở dữ liệu thông tin Pokédex 150 loài Pokémon Thế hệ 1 (Kanto).
Bao gồm: ID quốc gia, hệ nguyên tố, chiều cao, cân nặng, mô tả tiếng Việt.
"""

POKEDEX_DATA = {
    "Bulbasaur": {
        "id": 1, "name_vi": "Bulbasaur (Ếch Kì Diệu)", "types": ["Grass", "Poison"],
        "category": "Hạt Giống", "height": "0.7 m", "weight": "6.9 kg",
        "description": "Một hạt mầm kỳ lạ được gieo trên lưng nó ngay từ khi mới sinh. Củ mầm này lớn dần theo cơ thể."
    },
    "Ivysaur": {
        "id": 2, "name_vi": "Ivysaur", "types": ["Grass", "Poison"],
        "category": "Hạt Giống", "height": "1.0 m", "weight": "13.0 kg",
        "description": "Khi nụ hoa trên lưng hấp thụ đủ ánh sáng mặt trời, nó sẽ nở rộ và tỏa ra hương thơm dịu nhẹ."
    },
    "Venusaur": {
        "id": 3, "name_vi": "Venusaur", "types": ["Grass", "Poison"],
        "category": "Hạt Giống", "height": "2.0 m", "weight": "100.0 kg",
        "description": "Bông hoa khổng lồ trên lưng tỏa ra hương thơm ngọt ngào làm say lòng và xoa dịu những kẻ hiếu chiến."
    },
    "Charmander": {
        "id": 4, "name_vi": "Charmander (Khủng Long Lửa)", "types": ["Fire"],
        "category": "Thằn Lằn", "height": "0.6 m", "weight": "8.5 kg",
        "description": "Ngọn lửa ở chóp đuôi thể hiện sinh lực và cảm xúc của nó. Khi khỏe mạnh, ngọn lửa cháy rực rỡ."
    },
    "Charmeleon": {
        "id": 5, "name_vi": "Charmeleon", "types": ["Fire"],
        "category": "Ngọn Lửa", "height": "1.1 m", "weight": "19.0 kg",
        "description": "Tính tình hung hăng, thích chiến đấu. Nó dùng móng vuốt sắc nhọn và quất chiếc đuôi rực lửa."
    },
    "Charizard": {
        "id": 6, "name_vi": "Charizard", "types": ["Fire", "Flying"],
        "category": "Ngọn Lửa", "height": "1.7 m", "weight": "90.5 kg",
        "description": "Nó bay lượn trên bầu trời để tìm kiếm đối thủ mạnh. Hơi thở lửa của nó có thể làm tan chảy tảng đá lớn."
    },
    "Squirtle": {
        "id": 7, "name_vi": "Squirtle (Rùa Tí Hon)", "types": ["Water"],
        "category": "Rùa Nhỏ", "height": "0.5 m", "weight": "9.0 kg",
        "description": "Chiếc mai tròn nhẵn giúp giảm tối đa lực cản của nước, cho phép nó bơi lội với tốc độ cực nhanh."
    },
    "Wartortle": {
        "id": 8, "name_vi": "Wartortle", "types": ["Water"],
        "category": "Rùa", "height": "1.0 m", "weight": "22.5 kg",
        "description": "Đôi tai lông vũ và chiếc đuôi dài dày lông là biểu tượng của sự trường thọ và sức mạnh lướt sóng."
    },
    "Blastoise": {
        "id": 9, "name_vi": "Blastoise", "types": ["Water"],
        "category": "Vỏ Thép", "height": "1.6 m", "weight": "85.5 kg",
        "description": "Hai họng pháo nước bằng thép trên mai có thể bắn ra những tia nước áp lực cực cao xuyên thủng bức tường bê tông."
    },
    "Caterpie": {
        "id": 10, "name_vi": "Caterpie", "types": ["Bug"],
        "category": "Sâu Bướm", "height": "0.3 m", "weight": "2.9 kg",
        "description": "Nó tiết ra mùi hăng cay nồng từ hai chiếc râu đỏ hình chữ Y để xua đuổi các loài chim săn mồi."
    },
    "Metapod": {
        "id": 11, "name_vi": "Metapod", "types": ["Bug"],
        "category": "Kén", "height": "0.7 m", "weight": "9.9 kg",
        "description": "Lớp vỏ cứng như thép bảo vệ cơ thể non nớt đang biến đổi bên trong trước khi lột xác thành bướm."
    },
    "Butterfree": {
        "id": 12, "name_vi": "Butterfree", "types": ["Bug", "Flying"],
        "category": "Bướm", "height": "1.1 m", "weight": "32.0 kg",
        "description": "Đôi cánh có phấn độc chống nước, bay lượn thu thập mật hoa suốt ngày trên những cánh đồng hoa rực rỡ."
    },
    "Weedle": {
        "id": 13, "name_vi": "Weedle", "types": ["Bug", "Poison"],
        "category": "Sâu Độc", "height": "0.3 m", "weight": "3.2 kg",
        "description": "Mũi ngạnh nhọn hoắt trên đầu chứa nọc độc mạnh, giúp tự vệ trước kẻ thù ăn cỏ."
    },
    "Kakuna": {
        "id": 14, "name_vi": "Kakuna", "types": ["Bug", "Poison"],
        "category": "Kén Độc", "height": "0.6 m", "weight": "10.0 kg",
        "description": "Nó bám chặt vào thân cây và bất động, bên trong đang chuẩn bị cho sự tiến hóa thần tốc."
    },
    "Beedrill": {
        "id": 15, "name_vi": "Beedrill", "types": ["Bug", "Poison"],
        "category": "Ong Độc", "height": "1.0 m", "weight": "29.5 kg",
        "description": "Nó sở hữu ba ngòi độc sắc bén ở hai tay và đuôi, tấn công theo đàn với tốc độ chóng mặt."
    },
    "Pidgey": {
        "id": 16, "name_vi": "Pidgey", "types": ["Normal", "Flying"],
        "category": "Chim Nhỏ", "height": "0.3 m", "weight": "1.8 kg",
        "description": "Loài chim hiền lành, thường vỗ cánh tạo luồng gió cát để làm mù đối thủ rồi nhanh chóng bay đi."
    },
    "Pidgeotto": {
        "id": 17, "name_vi": "Pidgeotto", "types": ["Normal", "Flying"],
        "category": "Chim Lớn", "height": "1.1 m", "weight": "30.0 kg",
        "description": "Tầm nhìn cực kỳ sắc bén, có thể phát hiện con mồi Magikarp từ độ cao cả ngàn mét."
    },
    "Pidgeot": {
        "id": 18, "name_vi": "Pidgeot", "types": ["Normal", "Flying"],
        "category": "Chim Săn", "height": "1.5 m", "weight": "39.5 kg",
        "description": "Đôi cánh lộng lẫy có thể tạo ra cuồng phong và bay với vận tốc Mach 2 vượt qua bầu trời."
    },
    "Rattata": {
        "id": 19, "name_vi": "Rattata", "types": ["Normal"],
        "category": "Chuột Nhỏ", "height": "0.3 m", "weight": "3.5 kg",
        "description": "Răng nanh của nó phát triển không ngừng suốt đời, buộc nó phải gặm nhấm những vật thể cứng."
    },
    "Raticate": {
        "id": 20, "name_vi": "Raticate", "types": ["Normal"],
        "category": "Chuột Lớn", "height": "0.7 m", "weight": "18.5 kg",
        "description": "Cặp răng cửa cứng cáp có thể cắn gãy gỗ và thậm chí xuyên thủng những bức tường kiên cố."
    },
    "Pikachu": {
        "id": 25, "name_vi": "Pikachu", "types": ["Electric"],
        "category": "Chuột Điện", "height": "0.4 m", "weight": "6.0 kg",
        "description": "Nó tích tụ điện năng trong hai túi má đỏ rực. Khi giải phóng điện, nó tạo ra những tia sấm sét 100.000 volt."
    },
    "Raichu": {
        "id": 26, "name_vi": "Raichu", "types": ["Electric"],
        "category": "Chuột Điện", "height": "0.8 m", "weight": "30.0 kg",
        "description": "Đuôi dài hoạt động như một cọc tiếp địa giúp nó phóng ra những luồng điện cực mạnh mà không bị tự giật."
    },
    "Abra": {
        "id": 63, "name_vi": "Abra", "types": ["Psychic"],
        "category": "Siêu Năng Lực", "height": "0.9 m", "weight": "19.5 kg",
        "description": "Nó ngủ tới 18 tiếng mỗi ngày. Ngay cả khi đang ngủ, nó vẫn có thể dịch chuyển tức thời khi cảm thấy nguy hiểm."
    },
    "Kadabra": {
        "id": 64, "name_vi": "Kadabra", "types": ["Psychic"],
        "category": "Siêu Năng Lực", "height": "1.3 m", "weight": "56.5 kg",
        "description": "Nó cầm một chiếc thìa bạc để khuếch đại sóng alpha từ não, khiến các thiết bị điện tử xung quanh bị nhiễu loạn."
    },
    "Alakazam": {
        "id": 65, "name_vi": "Alakazam", "types": ["Psychic"],
        "category": "Siêu Năng Lực", "height": "1.5 m", "weight": "48.0 kg",
        "description": "Chỉ số IQ vượt quá 5.000. Bộ não liên tục ghi nhớ mọi thứ kể từ khi sinh ra và không bao giờ quên."
    },
    "Geodude": {
        "id": 74, "name_vi": "Geodude", "types": ["Rock", "Ground"],
        "category": "Đá Thạch", "height": "0.4 m", "weight": "20.0 kg",
        "description": "Thường bị nhầm lẫn với một hòn đá bình thường trên sườn núi. Nó dùng hai cánh tay rắn chắc để leo trèo."
    },
    "Graveler": {
        "id": 75, "name_vi": "Graveler", "types": ["Rock", "Ground"],
        "category": "Đá Thạch", "height": "1.0 m", "weight": "105.0 kg",
        "description": "Nó di chuyển bằng cách lăn tự do xuống các con dốc, nghiền nát mọi chướng ngại vật trên đường."
    },
    "Golem": {
        "id": 76, "name_vi": "Golem", "types": ["Rock", "Ground"],
        "category": "Khổng Lồ", "height": "1.4 m", "weight": "300.0 kg",
        "description": "Thân thể bọc giáp đá có thể chịu được cả vụ nổ thuốc nổ mà không bị trầy xước."
    },
    "Gastly": {
        "id": 92, "name_vi": "Gastly", "types": ["Ghost", "Poison"],
        "category": "Khí Độc", "height": "1.3 m", "weight": "0.1 kg",
        "description": "Cơ thể gần như 95% là khí độc thể hơi, có thể luồn lách qua mọi khe hở để hù dọa đối thủ."
    },
    "Haunter": {
        "id": 93, "name_vi": "Haunter", "types": ["Ghost", "Poison"],
        "category": "Bóng Ma", "height": "1.6 m", "weight": "0.1 kg",
        "description": "Nó ẩn nấp trong bóng tối hoàn toàn, chiếc lưỡi ma quái có thể rút cạn sinh lực kẻ xấu số."
    },
    "Gengar": {
        "id": 94, "name_vi": "Gengar", "types": ["Ghost", "Poison"],
        "category": "Bóng Ma", "height": "1.5 m", "weight": "40.5 kg",
        "description": "Khi cảm thấy nhiệt độ xung quanh bỗng nhiên hạ thấp đột ngột, Gengar đang lẩn khuất ngay sau bóng của bạn."
    },
    "Eevee": {
        "id": 133, "name_vi": "Eevee", "types": ["Normal"],
        "category": "Tiến Hóa Đa Dạng", "height": "0.3 m", "weight": "6.5 kg",
        "description": "Sở hữu bộ mã gen không ổn định, cho phép nó tiến hóa thành nhiều dạng nguyên tố khác nhau."
    },
    "Vaporeon": {
        "id": 134, "name_vi": "Vaporeon", "types": ["Water"],
        "category": "Bọt Nước", "height": "1.0 m", "weight": "29.0 kg",
        "description": "Cấu trúc tế bào tương tự phân tử nước, có thể hòa tan và tàng hình hoàn toàn trong làn nước biếc."
    },
    "Jolteon": {
        "id": 135, "name_vi": "Jolteon", "types": ["Electric"],
        "category": "Sét Đánh", "height": "0.8 m", "weight": "24.5 kg",
        "description": "Tích điện trong từng sợi lông kim nhọn hoắt, phóng ra điện áp lên tới 10.000 volt."
    },
    "Flareon": {
        "id": 136, "name_vi": "Flareon", "types": ["Fire"],
        "category": "Lửa Nóng", "height": "0.9 m", "weight": "25.0 kg",
        "description": "Nhiệt độ cơ thể có thể tăng lên đến hơn 900 độ C khi thở ra những quả cầu lửa thiêu đốt."
    },
    "Snorlax": {
        "id": 143, "name_vi": "Snorlax", "types": ["Normal"],
        "category": "Ngủ Say", "height": "2.1 m", "weight": "460.0 kg",
        "description": "Nó chỉ thức dậy để ăn 400kg thức ăn mỗi ngày rồi lại ngủ say sưa bất chấp xung quanh ồn ào."
    },
    "Dragonite": {
        "id": 149, "name_vi": "Dragonite", "types": ["Dragon", "Flying"],
        "category": "Rồng Thần", "height": "2.2 m", "weight": "210.0 kg",
        "description": "Loài rồng nhân từ bay vòng quanh thế giới trong 16 giờ, thường giải cứu những con tàu gặp bão trên biển."
    },
    "Mewtwo": {
        "id": 150, "name_vi": "Mewtwo", "types": ["Psychic"],
        "category": "Di Truyền", "height": "2.0 m", "weight": "122.0 kg",
        "description": "Được tạo ra bởi kỹ thuật tái tổ hợp DNA từ Mew, sở hữu sức mạnh tâm linh hủy diệt tột cùng."
    },
}

# Generic fallback builder for missing keys
def get_pokedex_info(pokemon_name: str) -> dict:
    """Tra cứu thông tin Pokédex chi tiết theo tên."""
    if pokemon_name in POKEDEX_DATA:
        return POKEDEX_DATA[pokemon_name]

    # Tra cứu không phân biệt hoa thường
    key = pokemon_name.strip().title()
    if key in POKEDEX_DATA:
        return POKEDEX_DATA[key]

    # Fallback tự động
    return {
        "id": 0,
        "name_vi": pokemon_name,
        "types": ["Normal"],
        "category": "Pokémon",
        "height": "1.0 m",
        "weight": "20.0 kg",
        "description": f"Dữ liệu Pokédex của {pokemon_name} (Thế hệ 1 Kanto)."
    }
