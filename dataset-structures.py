# All Data must look this format
# Need to convert "text" to string format
data_format = {
    "doc_id": "books_de_1",
    "language": "de",
    "source": "books",
    "title": "Künstliche Intelligenz",
    "text": (
        "Künstliche Intelligenz ist ein Teilgebiet der Informatik, das sich "
        "mit der Automatisierung intelligenten Verhaltens befasst.\n\n" * 15
    ),
    "url": "https://books.example.com/ki"
}



# The Dataset look like this for MLQA
# location - .\data\mlqa\dev\dev-context-ar-question-ar.json
mlqa = {
    "version": 1.0, 
    "data": [
        {
            'title': '2014年冬季奧林匹克運動會冰壺比賽', 
            'paragraphs': [
                {
                    'context': '本届冬奥会冰壶比赛参加资格有两种办法可以取得。各国家或地区可以通过2012年和2013年的世界冰壶锦标赛，也可以通过2013年12月举办的一次冬奥会资格赛来取得资格。七个国家通过两届世锦赛积分之和来获得资格，两个国家则通过冬奥会资格赛。作为主办国，俄罗斯自动获得参赛资格，这样就确定了冬奥会冰壶比赛的男女各十支参赛队伍。', 
                    'qas': [
                        {
                            'id': 'b08184972e38a79c47d01614aa08505bb3c9b680', 
                            'question': 'रूस ने कितनी टीमों के लिए क्वालीफाई किया?\\n', 
                            'answers': [
                                {
                                    'text': '十支', 
                                    'answer_start': 153
                                }
                            ]
                        }
                    ]
                },
                {
                    'context': ...,
                    'qas': [
                        {
                            ...
                        }
                    ]
                },
            ]
        },
        {
            'title': "...",
            'paragraphs': [
                ...
            ]
        }
    ]
}


# The only file for MLQA - with 10,000 more examples
# location - .\data\mkqa\ext\mkqa.jsonl
mkqa = {
    "query": "how long did it take the twin towers to be built",
    "answers": {
        "en": [
            {
                "type": "number_with_unit", 
                "text": "11.0 years", 
                "aliases": ["11 years"]
            }
        ], 
        "no": [{"type": "number_with_unit", "text": "11.0 År", "aliases": ["11 År"]}], 
        "ru": [{"type": "number_with_unit", "text": "11.0 лет", "aliases": ["11 лет"]}],
        "hu": [{"type": "number_with_unit", "text": "11.0 esztendő", "aliases": ["11 esztendő"]}], 
        "tr": [{"type": "number_with_unit", "text": "11.0 yıl", "aliases": ["11 yıl"]}], 
        "ms": [{"type": "number_with_unit", "text": "11.0 tahun", "aliases": ["11 tahun"]}], 
        "ja": [{"type": "number_with_unit", "text": "11.0 年", "aliases": ["11 年"]}], 
        "sv": [{"type": "number_with_unit", "text": "11.0 årtal", "aliases": ["11 årtal"]}], 
        "it": [{"type": "number_with_unit", "text": "11.0 anno", "aliases": ["11 anno"]}], 
        "pl": [{"type": "number_with_unit", "text": "11.0 rok", "aliases": ["11 rok"]}], 
        "ar": [{"type": "number_with_unit", "text": "11.0 سنة", "aliases": ["11 سنة"]}], 
        "th": [{"type": "number_with_unit", "text": "11.0 ปี", "aliases": ["11 ปี"]}], 
        "km": [{"type": "number_with_unit", "text": "11.0 years", "aliases": ["11 years"]}], 
        "nl": [{"type": "number_with_unit", "text": "11.0 jaar", "aliases": ["11 jaar"]}], 
        "ko": [{"type": "number_with_unit", "text": "11.0 연도", "aliases": ["11 연도"]}], 
        "es": [{"type": "number_with_unit", "text": "11.0 año terrestre", "aliases": ["11 año terrestre"]}], 
        "de": [{"type": "number_with_unit", "text": "11.0 Jahr", "aliases": ["11 Jahr"]}], 
        "pt": [{"type": "number_with_unit", "text": "11.0 ano", "aliases": ["11 ano"]}], 
        "vi": [{"type": "number_with_unit", "text": "11.0 năm", "aliases": ["11 năm"]}], 
        "fr": [{"type": "number_with_unit", "text": "11.0 années", "aliases": ["11 années"]}], 
        "he": [{"type": "number_with_unit", "text": "11.0 שנה", "aliases": ["11 שנה"]}], 
        "da": [{"type": "number_with_unit", "text": "11.0 år", "aliases": ["11 år"]}], 
        "fi": [{"type": "number_with_unit", "text": "11.0 vuosi", "aliases": ["11 vuosi"]}], 
        "zh_cn": [{"type": "number_with_unit", "text": "11.0 年份", "aliases": ["11 年份"]}], 
        "zh_tw": [{"type": "number_with_unit", "text": "11.0 年份", "aliases": ["11 年份"]}], 
        "zh_hk": [{"type": "number_with_unit", "text": "11.0 年份", "aliases": ["11 年份"]}]
    }, 
    "queries": {
        "tr": "ikiz kulelerin inşa edilmesi ne kadar sürdü", "hu": "mennyi ideig épültek az ikertornyok?", 
        "zh_hk": "建造twin towers用了多長時間", "nl": "hoelang duurde het om de twin towers te bouwen", 
        "ms": "menara berkembar petronas mengambil masa berapa lama untuk siap dibina", 
        "zh_cn": "世贸双塔建造用时多长", 
        "ja": "ツインタワーが建てられるまでどの位の時間がかかりましたか", 
        "de": "Wie lange dauerte es, um die Twin Towers zu bauen?", 
        "ru": "как долго строились башни-близнецы", 
        "pl": "ile czasu zajęło zbudowanie bliźniaczych wież", 
        "fi": "kuinka pitkään kaksoistorneja rakennettiin", 
        "pt": "quanto tempo levou para as torres gêmeas serem construídas", 
        "km": "តើវាត្រូវចំណាយពេលប៉ុន្មានដើម្បីសាងសង់ប៉មភ្លោះ", 
        "it": "Quanto ci è voluto per costruire le torri gemelle", 
        "fr": "combien de temps a-t-il fallu pour construire les tours jumelles", 
        "he": "כמה זמן לקח לבנות את מגדלי התאומים", 
        "vi": "Tòa tháp đôi được xây dựng trong bao lâu?", 
        "zh_tw": "世界貿易中心花了多久蓋好", 
        "no": "hvor lang tid tok det å bygge tvillingtårnene", 
        "da": "hvor lang tid tog det at bygge tvillinge tårnene", 
        "th": "ตึกคู่ใช้เวลาสร้างเท่าไหร่", 
        "sv": "hur lång tid tog det att bygga twin towers", 
        "es": "cuanto tardaron en construirse las torres gemelas", 
        "ar": ": كم من الوقت استغرق بناء البرجين التوأمين", 
        "en": "how long did it take the twin towers to be built", 
        "ko": "쌍둥이 빌딩이 지어지기 까지 얼마나 걸려"
    }, 
    "example_id": 3051930912491995402
}
mkqa_2 = {
    "query": "who sang the song you're my everything", 
    "answers": {
        "en": [
            {
                "type": "entity",
                "entity": "Q1833753", 
                "text": "Santa Esmeralda",
                "aliases": []
            },
            {
                "type": "entity", "entity": "Q846373", 
                "text": "The Temptations", 
                "aliases": ["Temptations"]
            }
        ], 
        "no": [{"type": "entity", "text": "Santa Esmeralda", "entity": "Q1833753"}, {"type": "entity", "text": "The Temptations", "entity": "Q846373"}], 
        "ru": [{"type": "entity", "entity": "Q1833753", "text": "Santa Esmeralda", "aliases": []}, {"type": "entity", "entity": "Q846373", "text": "The Temptations", "aliases": ["The Temps", "The Tempts"]}], 
        "hu": [{"type": "entity", "entity": "Q846373", "text": "The Temptations", "aliases": []}, {"type": "entity", "text": "Santa Esmeralda", "entity": "Q1833753"}], 
        "tr": [{"type": "entity", "entity": "Q1833753", "text": "Santa Esmeralda", "aliases": []}, {"type": "entity", "entity": "Q846373", "text": "The Temptations", "aliases": []}], 
        "ms": [{"type": "entity", "entity": "Q846373", "text": "The Temptations", "aliases": []}, {"type": "entity", "text": "Santa Esmeralda", "entity": "Q1833753"}, {"type": "entity", "text": "The Temptations", "entity": "Q846373"}], 
        "ja": [{"type": "entity", "entity": "Q1833753", "text": "サンタ・エスメラルダ", "aliases": []}, {"type": "entity", "entity": "Q846373", "text": "テンプテーションズ", "aliases": ["ザ・テンプテーションズ"]}], "sv": [{"type": "entity", "entity": "Q846373", "text": "The Temptations", "aliases": ["Temptations"]}, {"type": "entity", "text": "Santa Esmeralda", "entity": "Q1833753"}], "it": [{"type": "entity", "entity": "Q1833753", "text": "Santa Esmeralda", "aliases": []}, {"type": "entity", "entity": "Q846373", "text": "The Temptations", "aliases": ["Temptations"]}], "pl": [{"type": "entity", "entity": "Q846373", "text": "The Temptations", "aliases": []}, {"type": "entity", "text": "Santa Esmeralda", "entity": "Q1833753"}], "ar": [{"type": "entity", "text": "سانتا إزميرالدا", "entity": "Q1833753"}, {"type": "entity", "text": "The Temptations", "entity": "Q846373"}], "th": [{"type": "entity", "entity": "Q846373", "text": "เดอะเทมป์เทชันส์", "aliases": ["The Temptations", "เดอะเท็มป์เทชันส์"]}, {"type": "entity", "text": "Santa Esmeralda", "entity": "Q1833753"}], "km": [{"type": "entity", "text": "ซานตาเอสเมรัลดา", "entity": "Q1833753"}, {"type": "entity", "text": "ឋេ ដេម្ព្ត​តិឱន្ស្", "entity": "Q846373"}], "nl": [{"type": "entity", "entity": "Q1833753", "text": "Santa Esmeralda", "aliases": []}, {"type": "entity", "entity": "Q846373", "text": "The Temptations", "aliases": ["Temptations"]}], "ko": [{"type": "entity", "entity": "Q846373", "text": "템테이션스", "aliases": []}, {"type": "entity", "text": "산타 에스메랄다", "entity": "Q1833753"}], "es": [{"type": "entity", "entity": "Q1833753", "text": "Santa Esmeralda", "aliases": []}, {"type": "entity", "entity": "Q846373", "text": "The Temptations", "aliases": ["Temptations"]}], "de": [{"type": "entity", "entity": "Q1833753", "text": "Santa Esmeralda", "aliases": []}, {"type": "entity", "entity": "Q846373", "text": "The Temptations", "aliases": ["Temptations"]}], "pt": [{"type": "entity", "entity": "Q1833753", "text": "Santa Esmeralda", "aliases": []}, {"type": "entity", "entity": "Q846373", "text": "The Temptations", "aliases": ["Temptations"]}], "vi": [{"type": "entity", "entity": "Q846373", "text": "The Temptations", "aliases": []}, {"type": "entity", "text": "Santa Esmeralda", "entity": "Q1833753"}], "fr": [{"type": "entity", "entity": "Q1833753", "text": "Santa Esmeralda", "aliases": []}, {"type": "entity", "entity": "Q846373", "text": "The Temptations", "aliases": ["Temptations"]}], "he": [{"type": "entity", "entity": "Q846373", "text": "The Temptations", "aliases": ["הטמפטיישנס", "הפיתויים", "Temptations", "להקת הפיתויים"]}, {"type": "entity", "text": "Santa Esmeralda", "entity": "Q1833753"}], "da": [{"type": "entity", "entity": "Q846373", "text": "The Temptations", "aliases": []}, {"type": "entity", "text": "Santa Esmeralda", "entity": "Q1833753"}], "fi": [{"type": "entity", "entity": "Q1833753", "text": "Santa Esmeralda", "aliases": []}, {"type": "entity", "entity": "Q846373", "text": "The Temptations", "aliases": ["The Elgins", "The Primes", "The Tempts", "Temptations", "The Distants"]}], "zh_cn": [{"type": "entity", "entity": "Q846373", "text": "诱惑组合", "aliases": ["诱惑乐队"]}, {"type": "entity", "text": "Santa Esmeralda 乐队", "entity": "Q1833753"}], "zh_tw": [{"type": "entity", "text": "Santa Esmeralda", "entity": "Q1833753"}, {"type": "entity", "text": "誘惑合唱團 ", "entity": "Q846373"}], "zh_hk": [{"type": "entity", "text": "翡翠聖誕老人", "entity": "Q1833753"}, {"type": "entity", "text": "誘惑合唱團 ", "entity": "Q846373"}]
    },
    "queries": {"tr": "you're my everything şarkısını kim söylüyor", "hu": "Ki énekelte a You're My Everything című dalt?", "zh_hk": "邊個唱 you're my everything 一曲", "nl": "wie zong het nummer you're my everything", "ms": "siapa yang menyanyi lagu you're my everything", "zh_cn": "you're my everything这首歌是谁演唱的", "ja": "楽曲You're My Everythingを歌ったのは誰ですか", "de": "Wer hat das Lied you're my everything gesungen", "ru": "кто пел песню you're my everything", "pl": "kto śpiewał piosenkę you're my everything", "fi": "Kuka lauloi kappaleen you're my everything", "pt": "quem cantou a música você é meu tudo", "km": "ដែលច្រៀងចម្រៀង you're my everything", "it": "chi ha cantato la canzone you're my everything", "fr": "Qui chante la chanson You're my everything", "he": "מי שר את השיר you're my everything", "vi": "Ai hát bài you're my everything", "zh_tw": "誰演唱you're my everything這首歌", "no": "hvem sang sangen you're my everything", "da": "Hvem sang sangen you are my everything", "th": "ใครร้องเพลง you're my everything", "sv": "vem sjöng låten you're my everything", "es": "¿Quién cantaba la canción \"You're my everything\"?", "ar": "من الذي غنى أغنية يور ماي إيفريثنج", "en": "who sang the song you're my everything", "ko": "you're my everything노래는 누가 불렀나요"},
    "example_id": 510396804139041628
}
mkqa_3 = {"query": "pictures of samantha from sex and the city", "answers": {"en": [{"type": "unanswerable", "text": null}], "no": [{"type": "unanswerable", "text": null}], "ru": [{"type": "unanswerable", "text": null}], "hu": [{"type": "unanswerable", "text": null}], "tr": [{"type": "unanswerable", "text": null}], "ms": [{"type": "unanswerable", "text": null}], "ja": [{"type": "unanswerable", "text": null}], "sv": [{"type": "unanswerable", "text": null}], "it": [{"type": "unanswerable", "text": null}], "pl": [{"type": "unanswerable", "text": null}], "ar": [{"type": "unanswerable", "text": null}], "th": [{"type": "unanswerable", "text": null}], "km": [{"type": "unanswerable", "text": null}], "nl": [{"type": "unanswerable", "text": null}], "ko": [{"type": "unanswerable", "text": null}], "es": [{"type": "unanswerable", "text": null}], "de": [{"type": "unanswerable", "text": null}], "pt": [{"type": "unanswerable", "text": null}], "vi": [{"type": "unanswerable", "text": null}], "fr": [{"type": "unanswerable", "text": null}], "he": [{"type": "unanswerable", "text": null}], "da": [{"type": "unanswerable", "text": null}], "fi": [{"type": "unanswerable", "text": null}], "zh_cn": [{"type": "unanswerable", "text": null}], "zh_tw": [{"type": "unanswerable", "text": null}], "zh_hk": [{"type": "unanswerable", "text": null}]}, "queries": {"tr": "sex and the city den samantha nın fotoğrafları", "hu": "képek samantháról a Szex és New Yorkból", "zh_hk": "色慾都市裏面samantha的照片", "nl": "fotos van samantha uit sex and the city", "ms": "gambar samantha dari sex and the city", "zh_cn": "欲望都市中samantha 的照片", "ja": "sex and the city出演のsamanthaの写真", "de": "bilder von samantha aus sex and the city", "ru": "изображение Саманты из sex and the city", "pl": "zdjęcia samanty z Seksu w Wielkim Mieście", "fi": "kuvia sarjan sex and the city samanthasta", "pt": "fotos da samantha de sex and the city", "km": "រូបភាព Samantha មកពី \"សិចនិងទីក្រុង\"", "it": "Immagini di Samantha di \"sex and the city\"", "fr": "Photos de Samantha de Sex and the city", "he": "תמונות של סמנתה מסקס והעיר הגדולה", "vi": "hình ảnh của Samantha trong Sex And The City", "zh_tw": "慾望城市中莎曼珊的照片", "no": "bilder av samantha fra sex og singelliv", "da": "billeder af samantha fra sex and the city", "th": "รูปภาพของ samantha จากซีรีส์ sex and the city", "sv": "bilder på samantha från sex and the city", "es": "fotos de Samantha de sexo en nueva york", "ar": "صور لسامانثا في مسلسل سكس أند ذا سيتي", "en": "pictures of samantha from sex and the city", "ko": "세스 앤 더 시티에 사만다 사진"}, "example_id": -361906807922274081}
mkqa_4 = {"query": "what is the population of the state of maine", "answers": {"en": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "no": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "ru": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "hu": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "tr": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "ms": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "ja": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "sv": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "it": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "pl": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "ar": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "th": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "km": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "nl": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "ko": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "es": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "de": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "pt": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "vi": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "fr": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "he": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "da": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "fi": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "zh_cn": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "zh_tw": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}], "zh_hk": [{"type": "number", "text": "1338000.0", "aliases": ["1338000"]}]}, "queries": {"tr": "maine eyaletinin nüfusu kaçtır", "hu": "mi maine állam lakossága", "zh_hk": "緬因州的人口是多少", "nl": "wat is de populatie van de staat maine", "ms": "apakah populasi negeri maine", "zh_cn": "缅因州的人口是多少", "ja": "メイン州の人口は何人ですか", "de": "Wie groß ist die Bevölkerung des Staates Maine?", "ru": "какова численность населения штата Мэн", "pl": "jaka jest populacja stanu Maine", "fi": "mikä on maine:n osavaltion väkiluku", "pt": "qual a população do estado de maine", "km": "អ្វីដែលជាចំនួនប្រជាជននៃរដ្ឋនៃ Maine", "it": "Quanto è la popolazione dello stato del Maine", "fr": "Quelle est la population de l'état du Maine", "he": "מה אוכלוסיית מדינת מיין", "vi": "Dân số của bang maine là bao nhiêu", "zh_tw": "緬因州的人口是多少", "no": "hva er innbyggertallet i staten maine", "da": "hvor mange bor i staten maine", "th": "จำนวนประชากรของรัฐเมนคือเท่าใด", "sv": "vad är befolkningen i delstaten maine", "es": "¿Cuál es la población del estado de Maine?", "ar": "ما التعداد السكاني لولاية مين", "en": "what is the population of the state of maine", "ko": "메인 주의 인구수는 어"}, "example_id": -4713313337556490939}
mkqa_5 = {"query": "does full house beat a straight in poker", "answers": {"en": [{"type": "binary", "text": "yes"}], "no": [{"type": "binary", "text": "yes"}], "ru": [{"type": "binary", "text": "yes"}], "hu": [{"type": "binary", "text": "yes"}], "tr": [{"type": "binary", "text": "yes"}], "ms": [{"type": "binary", "text": "yes"}], "ja": [{"type": "binary", "text": "yes"}], "sv": [{"type": "binary", "text": "yes"}], "it": [{"type": "binary", "text": "yes"}], "pl": [{"type": "binary", "text": "yes"}], "ar": [{"type": "binary", "text": "yes"}], "th": [{"type": "binary", "text": "yes"}], "km": [{"type": "binary", "text": "yes"}], "nl": [{"type": "binary", "text": "yes"}], "ko": [{"type": "binary", "text": "yes"}], "es": [{"type": "binary", "text": "yes"}], "de": [{"type": "binary", "text": "yes"}], "pt": [{"type": "binary", "text": "yes"}], "vi": [{"type": "binary", "text": "yes"}], "fr": [{"type": "binary", "text": "yes"}], "he": [{"type": "binary", "text": "yes"}], "da": [{"type": "binary", "text": "yes"}], "fi": [{"type": "binary", "text": "yes"}], "zh_cn": [{"type": "binary", "text": "yes"}], "zh_tw": [{"type": "binary", "text": "yes"}], "zh_hk": [{"type": "binary", "text": "yes"}]}, "queries": {"tr": "full house pokerde straight'i yenebilir mi", "hu": "a full megveri a színsort a pókerben", "zh_hk": "在撲克比賽中, full house可以贏順子嗎", "nl": "verslaat full house een straat in poker", "ms": "adakah full house mengatasi straight dalam poker", "zh_cn": "满堂红能干过顺子吗", "ja": "ポーカーのフルハウスはストレートよりも強いか", "de": "schlägt Full House eine straight im Poker", "ru": "бьет ли фул хаус стрит в покере", "pl": "Czy full pokonuje strita w pokerze", "fi": "Voittaako täyskäsi suoran pokerissa", "pt": "full house vence um straight no poker?", "km": "តើផ្ទះពេញលេញនៅក្នុងបៀ Poker", "it": "il full batte una scala nel poker?", "fr": "est-ce que le full bat une quinte au poker", "he": "האם פול האוס גובר על סטרייט בפוקר", "vi": "cù lũ có thắng sảnh thùng trong bài poker không", "zh_tw": "在撲克牌遊戲中葫蘆比順子大嗎", "no": "slår fullt hus en straight i poker", "da": "slår fuldt hus en straight i poker", "th": "ฟูลเฮ้าส์ชนะสเตรทหรือไม่ในโป๊กเกอร์", "sv": "slår en kåk en stege i poker", "es": "¿El full es mejor que una escalera en el póquer?", "ar": "لا بيت كامل فاز مباشرة في لعبة البوكر", "en": "does full house beat a straight in poker", "ko": "포커에서 풀하우스가 스트레이트를 이기나요"}, "example_id": 8874324811832738126}
mkqa_6 = {"query": "when was while my guitar gently weeps written", "answers": {"en": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "no": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "ru": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "hu": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "tr": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "ms": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "ja": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "sv": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "it": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "pl": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "ar": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "th": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "km": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "nl": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "ko": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "es": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "de": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "pt": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "vi": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "fr": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "he": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "da": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "fi": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "zh_cn": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "zh_tw": [{"type": "date", "text": "1968", "aliases": ["1968"]}], "zh_hk": [{"type": "date", "text": "1968", "aliases": ["1968"]}]}, "queries": {"tr": "while my guitar gently weeps ne zaman yazıldı", "hu": "Mikor írták, hogy \"While My Guitar Gently Weeps\"?", "zh_hk": "while my guitar gently weeps是什麼時候寫的", "nl": "wanneer is while my guitar gently weeps geschreven", "ms": "bila while my guitar gently weeps ditulis", "zh_cn": "《While My Guitar Gently Weeps》是什么时候写的", "ja": "while my guitar gently weepsはいつに書かれたか", "de": "Wann wurde \"While My Guitar Gently Weeps\" geschrieben?", "ru": "когда написана while my guitar gently weeps", "pl": "kiedy powstała piosenka while my guitar gently weeps", "fi": "milloin kirjoitettiin kappale \"While My Guitar Gently Weeps\"", "pt": "quando foi escrito while my guitar gently weeps", "km": "តើ while my guitar gently weeps បានសរសេរនៅពេលណា", "it": "quando è stata scritta while my guitar gently weeps", "fr": "quand a été écrite while my guitar gently weeps", "he": "מתי נכתב השיר while my guitar gently weeps", "vi": "bài hát while my guitar gently weeps đã được viết khi nào", "zh_tw": "while my guitar gently weeps是什麼時候寫出來的", "no": "når ble while my guitar gently weeps skrevet", "da": "hvornår var while my guitar gently weeps written", "th": "เพลง while my guitar gently weeps แต่งขึ้นเมื่อปีใด", "sv": "när skrev while my guitar gently weeps", "es": "¿Cuándo se compuso la canción \"while my guitar gently weeps\"?", "ar": "متى تمت كتابة \"while my guitar gently weeps\"", "en": "when was while my guitar gently weeps written", "ko": "언제 while my guitar gently weeps"}, "example_id": 5759193192460133072}
mkqa_7 = {"query": "who has a red nose chip or dale", "answers": {"en": [{"type": "short_phrase", "text": "dale"}], "no": [{"type": "short_phrase", "text": "Snapp"}], "ru": [{"type": "short_phrase", "text": "дэйл"}], "hu": [{"type": "short_phrase", "text": "Dale"}], "tr": [{"type": "short_phrase", "text": "dale"}], "ms": [{"type": "short_phrase", "text": "Dale"}], "ja": [{"type": "short_phrase", "text": "デール"}], "sv": [{"type": "short_phrase", "text": "dal"}], "it": [{"type": "short_phrase", "text": "Dale"}], "pl": [{"type": "short_phrase", "text": " dołek"}], "ar": [{"type": "short_phrase", "text": "دايل"}], "th": [{"type": "short_phrase", "text": "หุบเขา"}], "km": [{"type": "short_phrase", "text": "Lembah"}], "nl": [{"type": "short_phrase", "text": "dale"}], "ko": [{"type": "short_phrase", "text": "데일"}], "es": [{"type": "short_phrase", "text": "valle"}], "de": [{"type": "short_phrase", "text": "Dale"}], "pt": [{"type": "short_phrase", "text": "Dale"}], "vi": [{"type": "short_phrase", "text": "dale"}], "fr": [{"type": "short_phrase", "text": "vallée"}], "he": [{"type": "short_phrase", "text": "העמק"}], "da": [{"type": "short_phrase", "text": "dale"}], "fi": [{"type": "short_phrase", "text": "Taku"}], "zh_cn": [{"type": "short_phrase", "text": "Dale"}], "zh_tw": [{"type": "short_phrase", "text": "蒂蒂"}], "zh_hk": [{"type": "short_phrase", "text": "Dale"}]}, "queries": {"tr": "Chip mi Dale mi kırmızı buruna sahip", "hu": "kinek van piros orra, chipnek vagy dalenek", "zh_hk": "誰有紅鼻子芯片或谷", "nl": "wie heeft een rode neus chip of dale", "ms": "siapa ada hidung merah dalam chip or dale", "zh_cn": "谁有 red nose chip or dale", "ja": "チップとデール、誰が赤い鼻ですか？", "de": "wer hat eine rote nase chip oder dale", "ru": "у кого красный носовой чип или дол", "pl": "Chip czy Dale ma czerwony nos", "fi": "kummalla on punainen nenä, tikulla vai takulla", "pt": "quem tem o nariz vermelho tico ou teco", "km": "អ្នកដែលមានបន្ទះឈីបច្រមុះក្រហមឬដាល់", "it": "chi ha il naso rosso chip o chop", "fr": "qui a le nez rouge Tic ou Tac", "he": "למי יש אף אדום צ'יפ או דייל", "vi": "chip hay dale ai có mũi đỏ", "zh_tw": "奇奇與蒂蒂誰有紅鼻子", "no": "hvem har rød nese chip eller dale", "da": "hvem har en rød næse chip eller chap", "th": "Chip หรือ Dale ที่มีจมูกสีแดง", "sv": "Vem i Piff och Puff har en röd näsa?", "es": "¿Quién tiene una nariz roja chip o \"dale\"?", "ar": "من صاحب الانف الاحمر تشيب اور ديل", "en": "who has a red nose chip or dale", "ko": "칩 또는 데일 중 누가 빨간 코인가요"}, "example_id": 5362196556471297450}
mkqa_8 = {"query": "what station is k love on the radio", "answers": {"en": [{"type": "long_answer", "text": null}], "no": [{"type": "long_answer", "text": null}], "ru": [{"type": "long_answer", "text": null}], "hu": [{"type": "long_answer", "text": null}], "tr": [{"type": "long_answer", "text": null}], "ms": [{"type": "long_answer", "text": null}], "ja": [{"type": "long_answer", "text": null}], "sv": [{"type": "long_answer", "text": null}], "it": [{"type": "long_answer", "text": null}], "pl": [{"type": "long_answer", "text": null}], "ar": [{"type": "long_answer", "text": null}], "th": [{"type": "long_answer", "text": null}], "km": [{"type": "long_answer", "text": null}], "nl": [{"type": "long_answer", "text": null}], "ko": [{"type": "long_answer", "text": null}], "es": [{"type": "long_answer", "text": null}], "de": [{"type": "long_answer", "text": null}], "pt": [{"type": "long_answer", "text": null}], "vi": [{"type": "long_answer", "text": null}], "fr": [{"type": "long_answer", "text": null}], "he": [{"type": "long_answer", "text": null}], "da": [{"type": "long_answer", "text": null}], "fi": [{"type": "long_answer", "text": null}], "zh_cn": [{"type": "long_answer", "text": null}], "zh_tw": [{"type": "long_answer", "text": null}], "zh_hk": [{"type": "long_answer", "text": null}]}, "queries": {"tr": "K-love radyoda hangi istasyonda/frekansta?", "hu": "Milyen állomáson van a K love a rádióban?", "zh_hk": "邊個電台播放 k love", "nl": "welk station is k love op de radio", "ms": "apa nama stesen radio k love di radio", "zh_cn": "k喜欢哪个电台节目", "ja": "ラジオの\"k love\"はどこの局か。", "de": "welcher sender ist k-love im radio", "ru": "на какой радиостанции идет k love", "pl": "Na jakiej stacji w radiu jest k love", "fi": "millä taajuudella k love -kanava on radiossa", "pt": "Quem estação de rádio é k love", "km": "តើស្ថានីយ៍វិទ្យុ k love ស្ថិតនៅលើវិទ្យុអ្វី?", "it": "Su quale frequenza è K love sulla radio", "fr": "quelle est la station K LOVE à la radio", "he": "איזו תחנה זה k love ברדיו", "vi": "kênh nào là k love on the radio", "zh_tw": "k love在廣播裡的哪個電台", "no": "hvilken frekvens er k-love på radioen", "da": "hvilken station er k love på radioen", "th": "สถานีอะไรคือ k love ในวิทยุ", "sv": "Vilken station är K-Love på radion", "es": "qué dial de radio tiene k love", "ar": "ما هي محطة كي لاف على الراديو", "en": "what station is k love on the radio", "ko": "K Love 라디오 주파수는 어디인가요"}, "example_id": -6348649568625217757}



# Taken from file ./xquad/train-v1.1.json
# location - .\data\xquad\train-v1.1.json
xquad = {
    'version': '1.1',
    "data": [
        {
            'title': 'University_of_Notre_Dame', 
            'paragraphs': [
                {
                    'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes".... ', 
                    'qas': [
                        {
                            'answers': [
                                {
                                    'answer_start': 515, 
                                    'text': 'Saint Bernadette Soubirous'
                                }
                            ], 
                            'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?', 
                            'id': '5733be284776f41900661182'
                        }, 
                        {'answers': [{'answer_start': 188, 'text': 'a copper statue of Christ'}], 'question': 'What is in front of the Notre Dame Main Building?', 'id': '5733be284776f4190066117f'}, 
                        {'answers': [{'answer_start': 279, 'text': 'the Main Building'}], 'question': 'The Basilica of the Sacred heart at Notre Dame is beside to which structure?', 'id': '5733be284776f41900661180'}, 
                        {'answers': [{'answer_start': 381, 'text': 'a Marian place of prayer and reflection'}], 'question': 'What is the Grotto at Notre Dame?', 'id': '5733be284776f41900661181'}, 
                        {'answers': [{'answer_start': 92, 'text': 'a golden statue of the Virgin Mary'}], 'question': 'What sits on top of the Main Building at Notre Dame?', 'id': '5733be284776f4190066117e'}
                    ]
                },
                {
                    'context': "As at most other universities, Notre Dame's students run a number of news media outlets. The nine student-run outlets include three newspapers, both a radio and television station, and several magazines and journals. ", 
                    'qas': [
                        {
                            'answers': [
                                {
                                    'answer_start': 248, 
                                    'text': 'September 1876'
                                }
                            ], 
                            'question': 'When did the Scholastic Magazine of Notre dame begin publishing?', 
                            'id': '5733bf84d058e614000b61be'
                        }, 
                        {'answers': [{'answer_start': 441, 'text': 'twice'}], 'question': "How often is Notre Dame's the Juggler published?", 'id': '5733bf84d058e614000b61bf'}, 
                        {'answers': [{'answer_start': 598, 'text': 'The Observer'}], 'question': 'What is the daily student paper at Notre Dame called?', 'id': '5733bf84d058e614000b61c0'}, 
                        {'answers': [{'answer_start': 126, 'text': 'three'}], 'question': 'How many student news papers are found at Notre Dame?', 'id': '5733bf84d058e614000b61bd'}, 
                        {'answers': [{'answer_start': 908, 'text': '1987'}], 'question': 'In what year did the student paper Common Sense begin publication at Notre Dame?', 'id': '5733bf84d058e614000b61c1'}
                    ]
                },
            ]
        },
        {
            "title": "Beyonc\u00e9", 
            "paragraphs": ...
        }
    ]
}
# location - data\xquad\xquad.de.json
xquad_2 = {
    "version": "1.1",
    "data": [
        {
            "title": "Force",
            "paragraphs": [
                {
                    "context": "The Panthers defense gave up just 308 points, ranking sixth in the league, while also leading the NFL in interceptions with 24 and boasting four Pro Bowl selections. Pro Bowl defensive tackle Kawann Short led the team in sacks with 11, while also forcing three fumbles and recovering two. Fellow lineman Mario Addison added 6\u00bd sacks. The Panthers line also featured veteran defensive end Jared Allen, a 5-time pro bowler who was the NFL's active career sack leader with 136, along with defensive end Kony Ealy, who had 5 sacks in just 9 starts. Behind them, two of the Panthers three starting linebackers were also selected to play in the Pro Bowl: Thomas Davis and Luke Kuechly. Davis compiled 5\u00bd sacks, four forced fumbles, and four interceptions, while Kuechly led the team in tackles (118) forced two fumbles, and intercepted four passes of his own. Carolina's secondary featured Pro Bowl safety Kurt Coleman, who led the team with a career high seven interceptions, while also racking up 88 tackles and Pro Bowl cornerback Josh Norman, who developed into a shutdown corner during the season and had four interceptions, two of which were returned for touchdowns.",
                    "qas": [
                        {
                            "answers": [
                                {
                                    "answer_start": 34,
                                    "text": "308"
                                }
                            ],
                            "id": "56beb4343aeaaa14008c925b",
                            "question": "How many points did the Panthers defense surrender?"
                        },
                        {
                            ...
                        }
                    ]
                },
                {
                    "context": "where  is the relevant cross-sectional area for the volume for which the stress-tensor is being calculated. This formalism includes pressure terms associated with forces that act normal to the cross-sectional area (the matrix diagonals of the tensor) as well as shear terms associated with forces that act parallel to the cross-sectional area (the off-diagonal elements). The stress tensor accounts for forces that cause all strains (deformations) including also tensile stresses and compressions.:133\u2013134:38-1\u201338-11",
                    "qas": [
                        {
                            "answers": [
                                {
                                    "answer_start": 376,
                                    "text": "stress tensor"
                                }
                            ],
                            "id": "5737a25ac3c5551400e51f51",
                            "question": "What causes strain in structures?"
                        },
                        {
                            ...
                        }
                    ]
                }
            ],
        }
    ],
}



# TyDiqa - each element in a separate line
# location - .\data\hf_TyDi_QA\english_TyDiQA_1.json
tydiqa = {
    "passage_answer_candidates": {
        "plaintext_start_byte":[5,378,956,1342,3245,3801,3838,4063],
        "plaintext_end_byte":[377,955,1317,3231,3792,3830,4057,4155]
    },
    "question_text":"Is Creole a pidgin of French?",
    "document_title":"French-based creole languages",
    "language":"english",
    "annotations":{
        "passage_answer_candidate_index":[1],
        "minimal_answers_start_byte":[-1],
        "minimal_answers_end_byte":[-1],
        "yes_no_answer":["YES"]
    },
    "document_plaintext": "\n\n\n\n\nPart of a series on theFrench language\nLangues d'oïl\nDialects\nCreoles\nFrancophonie\nHistory\nPhonological history\nOaths of Strasbourg\nOrdinance of Villers-Cotterêts\nAnglo-Norman\nGrammar\nAdverbs\nArticles and determiners\nPronouns (personal)\nVerbs (conjugationmorphology)\nOrthography\nAlphabet\nReforms\nCircumflex\nBraille\nPhonology\nElision\nLiaison\nAspirated h\nHelp:IPA\/Frenchvt\nA French creole, or French-based creole language, is a creole language (contact language with native speakers) for which French is the lexifier. Most often this lexifier is not modern French but rather a 17th-century koiné of French from Paris, the French Atlantic harbors, and the nascent French colonies. French-based creole languages are spoken natively by millions of people worldwide, primarily in the Americas and on archipelagos throughout the Indian Ocean. This article also contains information on French pidgin languages, contact languages that lack native speakers.\nThese contact languages are not to be confused with contemporary (non-creole) French language varieties spoken overseas in, for example, Canada (mostly in Quebec and the Maritime Provinces), the Canadian Prairie provinces, Louisiana, northern New England (Maine, New Hampshire, and Vermont). Haitian Creole is the most widely-spoken creole influenced by French.\nClassification\nAmericas\nVarieties with progressive aspect marker ape[1]\nHaitian Creole (Kreyòl ayisyen, locally called Creole) is a language spoken primarily in Haiti: the largest French-derived language in the world, with an estimated total of 12million fluent speakers. It is also the most-spoken creole language in the world and is based largely on 18th century French with influences from Portuguese, Spanish, English, Taíno, and West African languages.[2] It is an official language in Haiti.\nLouisiana Creole (Kréyol la Lwizyàn, locally called Kourí-Viní and Creole), the Louisiana creole language.\nVarieties with progressive aspect marker ka[3]\nAntillean Creole is a language spoken primarily in the francophone (and some of the anglophone) Lesser Antilles, such as Martinique, Guadeloupe, Îles des Saintes, Dominica, St. Lucia, Trinidad and Tobago and many other smaller islands. Although all of the creoles spoken on these islands are considered to be the same language, there are noticeable differences between the dialects of each island. Notably, the Creole spoken in the Eastern (windward) part of the island Saint-Barthélemy is spoken exclusively by a white population of European descent, imported into the island from Saint Kitts in 1648.\nDominican Creole French, Grenadian Creole French, Saint Lucian Creole French\nGuianese Creole is a language spoken in French Guiana, and to a lesser degree in Suriname and Guyana. It is closely related to Antillean Creole, but there are some noteworthy differences between the two.\nKaripúna, spoken in Brazil, mostly in the state of Amapá. It was developed by Amerindians, with possible influences from immigrants from neighboring French Guiana and French territories of the Caribbean and with a recent lexical adstratum from Portuguese.\nLanc-Patuá, spoken more widely in the state of Amapá, is a variety of the former, possibly the same language.\nIndian Ocean\nVarieties with progressive aspect marker ape[1] – subsumed under a common classification as Bourbonnais Creoles\nMauritian Creole, spoken in Mauritius (locally Kreol)\nAgalega Creole, spoken in Agaléga Islands\nChagossian Creole, spoken by the former population of the Chagos archipelago\nRéunion Creole, spoken in Réunion\nRodriguan Creole, spoken on the island of Rodrigues\nSeychellois Creole, spoken everywhere in the Seychelles and locally known as Kreol seselwa. It is the national language and shares official status with English and French.\nPacific\nTayo, spoken in New Caledonia\nAfrica\nPetit Mauresque or Little Moorish was spoken in North Africa\nFrançais Tirailleur, a Pidgin language [4] spoken in West Africa by soldiers in the French Colonial Army, approximately 1850–1960.\nCamfranglais in Cameroon\nAsia\nTây Bồi, Pidgin language spoken in former French Colonies in Indochina, primarily Vietnam\nSee also\n\nChiac\nMichif\nJoual\nNotes\n\n\n\n\n",
    "document_url":"https:\/\/en.wikipedia.org\/wiki\/French-based%20creole%20languages"
}




# CCNews - each element in a separate line
# location - .\data\hf_ccnews_extracted\de\batch_00000.json
ccnews = {
    "title": "25 Jahre Ötzi - und noch immer Geheimnisse", 
    "text": "Umhausen. Als Erika Simon in den Ötztaler Alpen beinahe über eine in Schmelzwasser liegende Leiche stolpert, glaubt sie, einen erfrorenen Skitourengeher entdeckt zu haben. Sie und ihr Mann Helmut benachrichtigen den Wirt einer nahe gelegenen Hütte. Wenige Tage darauf erfährt das Paar aus Nürnberg: Sie haben eine Mumie gefunden. Die Geschichte vom Ötzi geht um die Welt. 25 Jahre später kommen Erika Simon, Wissenschaftler und der Polizist, der den Ötzi damals ausgegraben hat, im Ötzi-Museum im österreichischen Umhausen zusammen.\nUnzählige Male hat Erika Simon ihre Geschichte erzählt. In den Wochen vor dem Jahrestag könnte sie fünf Interviews am Tag geben, erzählt sie. Ihr Mann ist vor mehr als zehn Jahren gestorben. Die 75-Jährigen erinnert sich genau an den 19. September 1991 - und an die Aufregung, die der Fund ausgelöst hat. Wissenschaftler sprechen vom \"Jahrhundertfund\". Eine fast unversehrte, knapp 5300 Jahre alte Mumie - älter als Pharao Tutanchamun.\nBeinahe wäre Ötzi als namenloser Bergsteiger begraben worden. Anton Koler war einer der Ersten an der Fundstelle. Der Polizist sollte den Toten gemeinsam mit dem Hüttenwirt bergen. Mit einem Pressluftmeißel versuchen sie, ihn aus dem Eis zu befreien. \"Er hat einen üblen Geruch verbreitet und war ledrig\", erinnert sich Koler. Bei der Leiche liegt ein Pickel, der ihm spanisch vorkommt, wie er sagt. Daneben \"Gerümpel\". Den Pickel schätzt Koler damals auf 150 Jahre und nimmt ihn mit zur Polizeiwache. Auch Ernst Schöpf, Bürgermeister von Sölden, erinnert sich gut. Bei der Bergung wird die Leiche an der Hüfte verletzt und ihr steif gefrorener Arm gebrochen - sonst hätte der Tote nicht in den Sarg gepasst, erzählt Schöpf.\nDass die Mumie nicht begraben wurde, sei auch Extrembergsteiger Reinhold Messner zu verdanken, der sich den Gletschermann anschaut. Seine spontane Einschätzung: Der stammt aus dem Mittelalter. Später werden Untersuchungen ergeben, dass Ötzi mehr als 5000 Jahre alt ist. Der Tote wird zunächst nach Innsbruck gebracht. Gut sechs Jahre befindet sich Ötzi dort in einer Klimazelle, die die Gegebenheiten im Eis simuliert. Höchstens 20 Minuten und nur alle vier Wochen nehmen die Forscher die Mumie für Arbeiten aus der Zelle.\nStreit um Fundort\nWährend die Forscher die Mumie zu entschlüsseln beginnen, entbrennt ein Streit: Wem gehört der Ötzi? Österreich oder Südtirol? Der Fundort am Tisenjoch wird neu vermessen. Dann steht fest: Der Ötzi lag 92,56 Meter von der Grenze entfernt auf italienischem Boden. Inzwischen ist Ötzi im Südtiroler Archäologiemuseum ausgestellt.\nWissenschaftler versuchen, alles über ihn herauszufinden: Was hat er gegessen? Woher stammt er? Wie ist er gestorben? Zehn Jahre nach dem Fund steht fest: Ötzi ist ermordet worden, hinterrücks mit einem Pfeil niedergestreckt. Die Mumie wird zum Kriminalfall. Raubmord schließen die Forscher aus, Ötzi hatte noch seinen wertvollen Kupferpickel bei sich. Auf der Flucht war er auch nicht. Denn seinem Mageninhalt nach hatte er kurz vor seinem Tod ausgiebig und fettreich gegessen. Die Wissenschaftler finden noch viel heraus: Laktose-Intoleranz, Zahnprobleme, Anlage zu Herz-Kreislauferkrankungen und zahlreiche Tätowierungen.",
    "url": "http://www.fnweb.de/nachrichten/25-jahre-otzi-und-noch-immer-geheimnisse-1.2965885", 
    "date_publish": "", "_source_parquet": "0000.parquet", "_source_rowgroup": 0, 
    "_global_idx": 0
}




# Wiki - dict_keys(['id', 'url', 'title', 'text']) - each element in a separate line
# location - .\data\hf_datasets_extracted\wikipedia_20231101_de\wikipedia_de_batch_00000.json
wiki = {
    "id":"76",
    "url":"https:\/\/de.wikipedia.org\/wiki\/Achsensprung%20%28Film%29",
    "title":"Achsensprung (Film)",
    "text":"Ein Achsensprung ist ein Filmschnitt, mit dem die Beziehungsachse der Figuren oder Gruppen übersprungen wird. Blickachsen\noder Beziehungsachsen zwischen den Akteuren untereinander oder dem Point of Interest des Protagonisten bilden eine gedachte Linie. Auf die Leinwand projiziert, stellt diese Linie eine „links-rechts-“ und „oben-unten-Beziehung“ zwischen den Akteuren dar. Mit Achsensprung bezeichnet man einen Schnitt, bei dem sich dieses Verhältnis umkehrt. Es wird zwischen Seitenachsensprung und dem Höhenachsensprung unterschieden. Letzterer wird als weniger desorientierend vom Zuschauer empfunden, da die Leinwand weniger hoch als breit ist. \nEin Achsensprung kann beim Zuschauer Desorientierung verursachen, da die Anordnung und Blickrichtung der Akteure im Frame sich relativ zum Zuschauer zu verändern scheint.\n\nAktionsachse (Handlungsachse)\nist die gedachte Linie, in deren Richtung sich die Handlung oder das Inertialsystem der Filmwelt bewegt. Bei einer Autofahrt zum Beispiel ist die Aktionsachse so stark, dass die Beziehungsachsen an Bedeutung verlieren. Die Orientierung bleibt trotz eventuellem Achsensprung bewahrt. Wenn man aus der Fahrerseite filmt, bewegt sich die Landschaft scheinbar von rechts nach links; filmt man aus der Beifahrerseite, bewegt sie sich scheinbar von links nach rechts. Diese Änderung der Bewegungsrichtung ist aber nicht irritierend. Analog werden zwei Autos, die bei einer Parallelmontage in die gleiche Richtung fahren (oft von links nach rechts, weil das unserer Leserichtung entspricht), als einander verfolgend wahrgenommen; wenn eines jedoch von links nach rechts und das andere von rechts nach links fährt, erwartet der Zuschauer einen Zusammenstoß.\n\nIm Continuity Editing des klassischen Hollywoodkinos wird der Achsensprung als Fehler betrachtet und dementsprechend vermieden. \n\nDer Grundsatz, Achsensprünge zu vermeiden, wird 180-Grad-Regel genannt.\n\nBewusster Achsensprung \nIn manchen Fällen kann ein bewusster Achsensprung auch Stilmittel sein, um beispielsweise Verwirrung oder einen Kippmoment zu symbolisieren; Stanley Kubrick wird in diesem Zusammenhang häufig genannt. In Werbespots werden Achsensprünge oft verwendet, um einen rasanten Effekt zu bewirken. Bekannt ist auch eine Szene aus Herr der Ringe, in welcher Sméagol mit sich selbst spricht. Da er mit den Schnitten wechselnd von der einen zur anderen Seite spricht (Achsensprung), entsteht der Eindruck zweier gleich aussehender Personen, womit der gespaltene Charakter der Figur unterstrichen wird.\n\nAchsenwechsel \nIm Gegensatz zum Achsensprung handelt es sich hierbei um eine Bewegung der Kamera (Steadicam oder einer Dollyfahrt) über die Achse oder um eine Änderung der Bewegungsachse bzw. der Blickrichtung der Figuren, wodurch eine neue Achse definiert wird. Der Achsenwechsel wird vom Zuschauer nicht als störend wahrgenommen, weil sich die Bewegung fließend vollzieht. Diese Bewegung wird mitunter auch als Crab bezeichnet. Außerdem kann ein Zwischenschnitt in eine Totale eine Achsenüberschreitung möglich machen, da so die räumliche Anordnung der Akteure für den Zuschauer deutlich wird, oder der Zwischenschnitt auf einen Closeup, da sich der Betrachter danach wieder neu räumlich orientiert.\n\nAchsen im Film \n Die Handlungsachse gibt die Hauptrichtung der Handlung an. Meist ist sie die Verbindung der Akteure, bei einer Fußballübertragung die Verbindung der Tore.\n Die Blickachse gibt die Blickrichtung und den Blickwinkel (Verhältnis zwischen der Höhen- und Seitenachse) der Figuren an. Bei Gesprächen ist darauf zu achten, dass sich die Figuren anschauen, was bedeutet, dass, wenn eine Figur in einem Bild nach links oben schaut, der Gesprächspartner im anderen Bild (Gegenschuss) nach rechts unten schaut. Diese Richtungen und die beiden Winkel sollten nicht verändert werden, solange sich die reale Blickrichtung bzw. der Standpunkt der Figuren nicht ändert.\n Die Kameraachse ist die „Blickrichtung“ der Kamera. Bei einer subjektiven Perspektive (Point of View) ist sie mit der Blickachse identisch.\n\nWeblinks \n Erklärvideo zu Achsensprung\nFilmgestaltung\nPostproduktion"
}