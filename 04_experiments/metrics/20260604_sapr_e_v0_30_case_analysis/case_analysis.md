# SAPR-E Minimal Rerank Case Analysis

## Summary

- n: 30
- category_counts: {'both_correct': 7, 'both_wrong': 19, 'improvement': 2, 'regression': 2}
- baseline_em_count: 9
- rerank_em_count: 9
- baseline_avg_f1: 0.43986772486772485
- rerank_avg_f1: 0.4126190476190476
- retrieval_changed_items: 22
- baseline_any_gold_items: 16
- rerank_any_gold_items: 16
- step_count: 50
- baseline_gold_hit_steps: 23
- rerank_gold_hit_steps: 29
- rerank_added_gold_steps: 11
- rerank_removed_gold_steps: 5
- both_gold_steps: 18

## improvement cases

| id | gold | baseline -> rerank | doc steps | gold item-hit | question |
|---|---|---|---|---|---|
| dev_3 | ['no'] |  -> No | 4/3 | True/True | Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood? |
| | | step 0 overlap 3 | B: ['Esma Sultan (daughter of Abdul Hamid I)', 'Esma Sultan', 'Esma Sultan Mansion'] | R: ['Esma Sultan Mansion', 'Esma Sultan (daughter of Abdul Hamid I)', 'Esma Sultan'] | gold B/R: ['Esma Sultan Mansion']/['Esma Sultan Mansion'] |
| | | step 1 overlap 2 | B: ['Holy Spirit Cathedral, Jataí', 'Esma Sultan', 'Laleli Mosque'] | R: ['Tayyare Apartments', 'Laleli Mosque', 'Esma Sultan'] | gold B/R: ['Laleli Mosque']/['Laleli Mosque'] |
| | | step 2 overlap 1 | B: ['Ortaköy', 'Ortaköy', 'Laleli Mosque'] | R: ['Tayyare Apartments', 'Laleli Mosque', 'Ortaköy, Çorum'] | gold B/R: ['Laleli Mosque']/['Laleli Mosque'] |
| dev_13 | ['no'] | yes -> No | 2/0 | True/False | Are Random House Tower and 888 7th Avenue both used for real estate? |
| | | step 0 overlap 0 | B: ['Pershing Square Capital Management', 'Random House Tower', 'Random House Studio'] | R: [] | gold B/R: ['Random House Tower']/[] |
| | | step 1 overlap 0 | B: ['Random House Tower', 'Random House Studio', 'Pershing Square Capital Management'] | R: [] | gold B/R: ['Random House Tower']/[] |

## regression cases

| id | gold | baseline -> rerank | doc steps | gold item-hit | question |
|---|---|---|---|---|---|
| dev_10 | ['Kansas Song'] | Kansas Song -> I'm a Jayhawk | 1/1 | False/True | What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas Ci |
| | | step 0 overlap 0 | B: ['Steven Ronald Jensen', 'Michelangelo Faggioli', "We're Not Gonna Take It (The Who song)"] | R: ['Kansas State University Marching Band', 'University of Kansas', "I'm a Jayhawk"] | gold B/R: []/['University of Kansas', 'University of Kansas'] |
| dev_20 | ['Pedro Rodríguez'] | Pedro Rodríguez -> Manuel Fittipaldi | 2/2 | True/True | Which other Mexican Formula One race car driver has held the podium besides the Force India driver born in 1990? |
| | | step 0 overlap 0 | B: ['Ricardo Rodríguez (racing driver)', 'BAR 007', 'Jorge Arteaga'] | R: ['Force India', 'Guadalajara', 'Formula One drivers from Mexico'] | gold B/R: []/['Formula One drivers from Mexico', 'Formula One drivers fro |
| | | step 1 overlap 1 | B: ['Formula One drivers from Mexico', 'Romain Grosjean', 'Gene Haas'] | R: ['Sergio Pérez', 'Formula One drivers from Mexico', 'Formula One drivers from Mexico'] | gold B/R: ['Formula One drivers from Mexico', 'Formula One drivers fro/['Formula One drivers from Mexico', 'Formula One drivers fro |

## both_wrong cases

| id | gold | baseline -> rerank | doc steps | gold item-hit | question |
|---|---|---|---|---|---|
| dev_1 | ['Chief of Protocol'] | United States ambassador to Ghana and to Czechoslovakia, and Chief of Protocol o -> United States Ambassador to Ghana | 2/2 | True/True | What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell? |
| | | step 0 overlap 1 | B: ['A Kiss for Corliss', 'Breathless (1960 film)', 'Kiss and Tell (1945 film)'] | R: ['Kiss and Tell (play)', 'Kiss and Tell (1945 film)', 'Meet Corliss Archer'] | gold B/R: ['Kiss and Tell (1945 film)']/['Kiss and Tell (1945 film)'] |
| | | step 1 overlap 1 | B: ['Shirley Temple', 'Shirley Temple', 'Jennifer Phang'] | R: ['Warsaw Pact invasion of Czechoslovakia', 'Shirley Temple', 'Shirley Temple'] | gold B/R: ['Shirley Temple', 'Shirley Temple']/['Shirley Temple', 'Shirley Temple'] |
| dev_2 | ['Animorphs'] | The Stormlight Archive -> The Stormlight Archive | 0/0 | False/False | What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and ali |
| dev_4 | ['Greenwich Village, New York City'] | New York City -> New York City | 1/1 | True/True | The director of the romantic comedy "Big Stone Gap" is based in what New York city? |
| | | step 0 overlap 3 | B: ['Matthew Goode', 'Big Stone Gap (film)', 'Great Eastern Conventions'] | R: ['Big Stone Gap (film)', 'Matthew Goode', 'Great Eastern Conventions'] | gold B/R: ['Big Stone Gap (film)']/['Big Stone Gap (film)'] |
| dev_5 | ['YG Entertainment'] | S.M. Entertainment -> S.M. Entertainment | 1/1 | False/False | 2014 S/S is the debut album of a South Korean boy group that was formed by who? |
| | | step 0 overlap 1 | B: ['Shinee', 'The First (album)', 'Base (EP)'] | R: ['The First (album)', 'Choi Min-ho (entertainer)', 'Inquilabi Communist Sangathan'] | gold B/R: []/[] |
| dev_7 | ['3,677 seated'] | 5,000 -> 5,000 | 0/0 | False/False | The arena where the Lewiston Maineiacs played their home games can seat how many people? |
| dev_8 | ['Terry Richardson'] | Annie Morton is younger than Terry Richardson (ice hockey). So the answer is Ter -> Annie Morton | 3/1 | True/True | Who is older, Annie Morton or Terry Richardson? |
| | | step 0 overlap 0 | B: ['The Mummy: The Animated Series', 'Alicia Morton', 'Annie Morton'] | R: ['Terry Richardson (rugby league)', 'Terry Richardson (ice hockey)', 'Terry Richardson'] | gold B/R: ['Annie Morton', 'Annie Morton']/['Terry Richardson'] |
| | | step 1 overlap 0 | B: ['Terry Richardson (ice hockey)', 'Terry Richardson (rugby league)', 'Terry Richardson'] | R: [] | gold B/R: ['Terry Richardson']/[] |
| | | step 2 overlap 0 | B: ['Terry Richardson (ice hockey)', 'The Bronze (film)', 'MLX Skates'] | R: [] | gold B/R: []/[] |
| dev_9 | ['yes'] | No -> No | 0/0 | False/False | Are Local H and For Against both from the United States? |
| dev_11 | ['David Weissman'] | David Diamond -> David Diamond and David Weissman | 1/1 | False/False | What screenwriter with credits for "Evolution" co-wrote a film starring Nicolas Cage and Téa Leoni? |
| | | step 0 overlap 2 | B: ['David Diamond (screenwriter)', 'Nicolas Cage', 'Gundala (film)'] | R: ['David Diamond (screenwriter)', 'Drive Angry', 'Gundala (film)'] | gold B/R: []/[] |
| dev_12 | ['1999'] |  ->  | 4/4 | False/False | What year did Guns N Roses perform a promo for a movie starring Arnold Schwarzenegger as a former New York Police detective? |
| | | step 0 overlap 1 | B: ['Arnold Schwarzenegger', 'who is the former body builder who became a film star and a governor', 'Arnold Schwarzenegger | R: ['Arnold Schwarzenegger', 'Arnold Schwarzenegger', 'Arnold Schwarzenegger'] | gold B/R: []/[] |
| | | step 1 overlap 0 | B: ['Not in This Lifetime... Tour', 'Live ?!*@ Like a Suicide', 'Appetite for Destruction Tour'] | R: ['Arnold Schwarzenegger', 'Arnold Schwarzenegger', 'Arnold Schwarzenegger'] | gold B/R: []/[] |
| | | step 2 overlap 0 | B: ['Not in This Lifetime... Tour', 'Appetite for Destruction Tour', "Guns N' Roses"] | R: ['Arnold Schwarzenegger', 'Arnold Schwarzenegger', 'Arnold Schwarzenegger'] | gold B/R: []/[] |
| dev_14 | ['from 1986 to 2013'] | 1986 to 2013 -> 1986 to 2013 | 1/1 | False/False | The football manager who recruited David Beckham managed Manchester United during what timeframe? |
| | | step 0 overlap 3 | B: ['when did sir alex ferguson join man utd', 'when did sir alex ferguson leave manchester united', 'History of Manchester | R: ['when did sir alex ferguson leave manchester united', 'when did sir alex ferguson join man utd', 'History of Manchester | gold B/R: []/[] |
| dev_15 | ['9,984'] | 2,907,289 -> 2,907,289 | 2/2 | True/True | Brown State Fishing Lake is in a country that has a population of how many inhabitants ? |
| | | step 0 overlap 1 | B: ['Fishing Lake', 'Brown State Fishing Lake', 'Brown State Fishing Lake'] | R: ['Brown State Fishing Lake', 'Brown State Fishing Lake', 'Brown State Fishing Lake'] | gold B/R: ['Brown State Fishing Lake']/['Brown State Fishing Lake'] |
| | | step 1 overlap 1 | B: ['Kansas', 'Wichita, Kansas', 'Kansas City, Missouri'] | R: ['Johnson County, Kansas', 'Kansas', 'McPherson County, Kansas'] | gold B/R: []/[] |
| dev_16 | ['the North Atlantic Conference'] | ECAC Hockey ->  | 1/4 | True/True | The Vermont Catamounts men's soccer team currently competes in a conference that was formerly known as what from 1988 to 1996? |
| | | step 0 overlap 1 | B: ["Vermont Catamounts men's ice hockey", "Vermont Catamounts men's soccer", "Vermont Catamounts men's soccer"] | R: ["Vermont Catamounts men's soccer", "Vermont Catamounts men's soccer", "Vermont Catamounts men's soccer"] | gold B/R: ["Vermont Catamounts men's soccer", "Vermont Catamounts men'/["Vermont Catamounts men's soccer", "Vermont Catamounts men' |
| | | step 1 overlap 0 | B: [] | R: ["Vermont Catamounts men's soccer", 'Vermont Catamounts football', "Vermont Catamounts men's soccer"] | gold B/R: []/["Vermont Catamounts men's soccer", "Vermont Catamounts men' |
| | | step 2 overlap 0 | B: [] | R: ["Vermont Catamounts men's soccer", 'Vermont Catamounts football', "Vermont Catamounts men's soccer"] | gold B/R: []/["Vermont Catamounts men's soccer", "Vermont Catamounts men' |
| dev_18 | ['1969 until 1974'] | 1969-1974 -> 1969-1974 | 2/2 | True/True | Roger O. Egeberg was Assistant Secretary for Health and Scientific Affairs during the administration of a president that served during what  |
| | | step 1 overlap 1 | B: ['who was the president of the united states in 1970', 'Presidency of Richard Nixon', 'Marvin Minoff'] | R: ['Presidency of Richard Nixon', 'Federal pardons in the United States', 'National Security Advisor (Nigeria)'] | gold B/R: []/[] |
| dev_19 | ['Robert Erskine Childers DSC'] | Henry Roth ->  | 1/3 | True/True | Which writer was from England, Henry Roth or Robert Erskine Childers? |
| | | step 0 overlap 1 | B: ['Henry Ling Roth', 'Henry Roth', 'Henry Lincoln'] | R: ['Henry Roth', 'Henry Roth', 'Henry Roth'] | gold B/R: ['Henry Roth']/['Henry Roth'] |
| | | step 1 overlap 0 | B: [] | R: ['Henry Roth', 'Henry Roth', 'Henry Roth'] | gold B/R: []/['Henry Roth'] |
| | | step 2 overlap 0 | B: [] | R: ['Henry Roth', 'Henry Roth', 'Henry Roth'] | gold B/R: []/['Henry Roth'] |
| dev_21 | ['Sonic'] | Bradley Michael Pierce -> James McAvoy | 3/4 | False/False | This singer of A Rather Blustery Day also voiced what hedgehog? |
| | | step 0 overlap 1 | B: ['The Tale of Despereaux (film)', 'Markku Pölönen', 'Ann Dowd'] | R: ['The Tale of Despereaux (film)', "a turtle's tale sammy's adventures cast", 'David Humphrey'] | gold B/R: []/[] |
| | | step 1 overlap 1 | B: ['The Tale of Despereaux (film)', 'Markku Pölönen', 'Ann Dowd'] | R: ['The Tale of Despereaux (film)', "a turtle's tale sammy's adventures cast", 'David Humphrey'] | gold B/R: []/[] |
| | | step 2 overlap 1 | B: ['The Tale of Despereaux (film)', 'Markku Pölönen', 'Ann Dowd'] | R: ['The Tale of Despereaux (film)', "a turtle's tale sammy's adventures cast", 'David Humphrey'] | gold B/R: []/[] |
| dev_22 | ['keyboard function keys'] | Remote Buddy or mira -> VLC media player | 1/1 | True/True | Aside from the Apple Remote, what other device can control the program Apple Remote was originally designed to interact with? |
| | | step 0 overlap 1 | B: ['Apple Remote', 'Apple Remote', 'ITunes Remote'] | R: ['Apple Remote', 'Apple Remote', 'Apple Remote'] | gold B/R: ['Apple Remote', 'Apple Remote']/['Apple Remote', 'Apple Remote'] |
| dev_23 | ['Badly Drawn Boy'] |  ->  | 3/2 | True/True | Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice?  |
| | | step 0 overlap 1 | B: ['Badly Drawn Boy', 'Ponderosa Twins Plus One', 'Badly Drawn Boy'] | R: ['Badly Drawn Boy', 'Badly Drawn Boy', 'Badly Drawn Boy'] | gold B/R: ['Badly Drawn Boy']/['Badly Drawn Boy'] |
| | | step 1 overlap 1 | B: ['Passive (song)', 'Blush (Wolf Alice EP)', 'Blush (Wolf Alice EP)'] | R: ['Wolf Alice', 'Wolf Alice', 'Blush (Wolf Alice EP)'] | gold B/R: []/['Wolf Alice'] |
| | | step 2 overlap 0 | B: ['Badly Drawn Boy', 'Ponderosa Twins Plus One', 'Draw-a-Person test'] | R: [] | gold B/R: ['Badly Drawn Boy']/[] |
| dev_25 | ['Barton Lee Hazlewood'] | Leiber and Stoller -> Leiber and Stoller | 0/0 | False/False | Who was the writer of These Boots Are Made for Walkin' and who died in 2007? |
| dev_29 | ['Arena of Khazan'] | Circle of Ice -> None | 1/4 | True/True | What is the name for the adventure in "Tunnels and Trolls", a game designed by Ken St. Andre? |
| | | step 0 overlap 1 | B: ['Tunnels &amp; Trolls', 'Troll and I', 'Circle of Ice'] | R: ['Tunnels & Trolls', 'Tunnels & Trolls', 'Tunnels &amp; Trolls'] | gold B/R: ['Tunnels &amp; Trolls']/['Tunnels &amp; Trolls'] |
| | | step 1 overlap 0 | B: [] | R: ['Tunnels & Trolls', 'Tunnels & Trolls', 'Tunnels &amp; Trolls'] | gold B/R: []/['Tunnels &amp; Trolls'] |
| | | step 2 overlap 0 | B: [] | R: ['Tunnels & Trolls', 'Tunnels & Trolls', 'Tunnels &amp; Trolls'] | gold B/R: []/['Tunnels &amp; Trolls'] |

## both_correct cases

| id | gold | baseline -> rerank | doc steps | gold item-hit | question |
|---|---|---|---|---|---|
| dev_0 | ['yes'] | yes -> yes | 0/0 | False/False | Were Scott Derrickson and Ed Wood of the same nationality? |
| dev_6 | ['Eenasul Fateh'] | Eenasul Fateh -> Eenasul Fateh | 2/1 | True/True | Who was known by his stage name Aladin and helped organizations improve their performance as a consultant? |
| | | step 0 overlap 1 | B: ['Lil Ru', 'Eenasul Fateh', 'Eenasul Fateh'] | R: ['Aladin & Alakadam', 'Eenasul Fateh', 'Eenasul Fateh'] | gold B/R: ['Eenasul Fateh']/['Eenasul Fateh'] |
| | | step 1 overlap 0 | B: ['Vocational Rehabilitation Act of 1973', 'Fred Luthans', 'Abul Fateh'] | R: [] | gold B/R: []/[] |
| dev_17 | ['yes'] | Yes -> Yes | 0/0 | False/False | Are Giuseppe Verdi and Ambroise Thomas both Opera composers ? |
| dev_24 | ["World's Best Goalkeeper"] | World's Best Goalkeeper -> World's Best Goalkeeper | 1/1 | True/True | What was the father of Kasper Schmeichel voted to be by the IFFHS in 1992? |
| | | step 0 overlap 1 | B: ['Peter Schmeichel', 'Peter Schmeichel', "Sommeren '92"] | R: ['Kasper Lorentzen', 'Kasper Lorentzen', 'Peter Schmeichel'] | gold B/R: ['Peter Schmeichel']/['Peter Schmeichel'] |
| dev_26 | ['1838'] | 1838 -> 1838 | 1/2 | True/True | The 2011–12 VCU Rams men's basketball team, led by third year head coach Shaka Smart, represented Virginia Commonwealth University which was |
| | | step 0 overlap 2 | B: ['Capital City Classic', "Governor's Foreign Language Academies", 'Virginia Commonwealth University'] | R: ['History of Virginia Commonwealth University', 'Virginia Commonwealth University', "Governor's Foreign Language Academi | gold B/R: ['Virginia Commonwealth University']/['Virginia Commonwealth University'] |
| | | step 1 overlap 0 | B: [] | R: ["2014–15 VCU Rams men's basketball team", "2004–05 VCU Rams men's basketball team", "2012–13 VCU Rams men's basketball  | gold B/R: []/[] |
| dev_27 | ['yes'] | yes -> yes | 0/0 | False/False | Are both Dictyosperma, and Huernia described as a genus? |
| dev_28 | ['Henry J. Kaiser'] | Henry J. Kaiser -> Henry J. Kaiser | 0/0 | False/False | Kaiser Ventures corporation was founded by an American industrialist who became known as the father of modern American shipbuilding? |