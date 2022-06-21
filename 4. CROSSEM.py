import tweepy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATASET = 'US_Election'

# Part - 1
# unique users should be obtained from output of Data Active Accounts
unique_users = {'Laura Ingraham': 18, 'Diamond and Silk®': 10, 'Will Saletan': 1, 'Bernie Sanders': 13, 'Ann Coulter': 5, 'Kellyanne Conway': 107, 'WikiLeaks': 100, 'USA NEWS': 16, 'Wiz Khalifa - Motivational Quotes': 1, 'Kurt Eichenwald': 17, 'Hillary Clinton': 365, 'OR Books': 1, 'Mike Pence': 21, 'Disability Action': 1, "Dinesh D'Souza": 7, 'The Associated Press': 3, 'White House Archived': 5, 'MARK SIMONE': 2, 'danny': 1, 'All Things Trump': 1, 'Scott Adams': 5, 'Paul Joseph Watson': 48, 'Jeanine Pirro': 7, 'knth': 1, 'Tim Kaine': 18, 'eric': 1, 'President Obama': 20, 'The New York Times': 5, 'Chelsea Clinton': 1, 'Marc Lotter': 1, 'Dan Scavino Jr.🇺🇸🦅': 20, 'Donald Trump Jr.': 43, 'Paul Krugman': 2, '🇺🇸ERIC BOLLING🇺🇸': 1, 'Eric Trump': 10, 'Tamara Keith': 1, 'Clint Smith': 1, 'George Takei': 4, 'Wayne Dupree Media, LLC ⭐️': 4, 'Ozzy': 1, 'Great America PAC': 1, 'Hillary for America': 2, 'Tim Young': 1, 'Liz Romanoff Silva': 1, 'Michael Keyes': 4, 'bailey': 1, 'Tony Schwartz': 3, 'Newsweek': 2, 'Dr. Jill Stein🌻': 8, 'Stephen Colbert': 3, 'James Woods': 12, 'Razor': 3, 'Stephen King': 2, 'Malik Obama': 2, 'Lou Dobbs': 12, 'Patton Oswalt': 1, 'HARLAN Z. HILL 🇺🇸': 9, 'Walter Cronkite': 3, 'Makada': 2, 'tyler oakley (gremlin era 👹)': 1, 'Marv 𓆉': 1, 'Cristina Laila': 2, 'Brian Stelter': 2, 'Pb': 1, 'RealVinnieJames': 2, 'CNN': 6, 'Melanie': 1, 'Dallas Morning News': 1, 'Jeb Bush': 1, 'Joy-Ann (Pro-Democracy) Reid 😷': 2, 'John doe': 2, 'Chris Hayes': 3, 'David Fahrenthold': 4, 'Newt Gingrich': 6, 'Chris Sacca 🇺🇸': 1, 'Chad Griffin': 1, 'Clinton Foundation': 3, 'Ted Cruz': 3, 'Mari Copeny': 1, 'paulova \uea00': 1, 'Joel Pollak': 1, 'Fox News': 14, 'Senator Harry Reid': 1, 'jacksfilms🌹': 1, 'Kathy Shelton': 2, 'Trevor 🇺🇸': 1, 'Yanik Dumont Baron': 1, 'The Wokest Numbersmuncher': 2, 'The Volatile Mermaid': 1, 'First Lady- Archived': 2, 'Linda Suhler, PhD': 3, 'BuzzFeed News': 2, 'Chris Hadfield': 1, 'Leonardo DiCaprio': 1, 'Breitbart News': 4, 'Nate Silver': 1, 'Colin Jones': 1, 'Judd Apatow 🇺🇦': 1, 'Joe Biden': 4, 'The Hill': 2, 'Wall Street Journal Opinion': 1, 'Washington Examiner': 4, 'DEPLORABLE DANI': 4, 'FOX & friends': 9, 'golden state blond 💙🏀💛': 1, 'Richard Weaving': 3, 'azcentral': 1, 'Maverick': 1, 'Daniel Dale': 5, 'Opinion by Tampa Bay Times': 1, 'Michael Cohen': 2, 'Ken Tremendous': 1, 'The Washington Post': 2, 'Felix Wu': 1, 'Enquirer': 1, 'Richard Hine': 1, 'J': 2, 'Varney & Co.': 1, 'Slate': 1, 'Business Insider': 1, 'Bruce Porter Jr.': 1, 'p': 1, 'Kambree': 1, 'Irma 🕊🤍': 1, 'Sebastian Gorka DrG': 2, 'Ivanka Trump': 3, 'Daniel Lin': 1, 'Bill Maher': 5, 'Tammy Duckworth': 1, 'John Podesta': 2, 'Frank Luntz': 6, 'LeBron James': 1, 'Brit Hume': 1, 'Dr. Marty Fox 🇺🇸': 1, 'Robby Mook': 1, 'Paul Myers': 1, 'Jason Miller': 2, 'Stolen Pokémon': 1, 'Lee Fang': 1, 'Oliver Chinyere': 1, 'Harold Itzkowitz': 1, 'Stefon on 2020': 1, 'Marlee Matlin': 1, 'Bloomberg Politics': 1, 'Chuck Woolery': 1, 'Peter Daou': 4, 'J Burton': 3, 'Every Voice': 1, 'Brian Fallon': 6, 'VP Biden (Archived)': 2, 'I Am Leah': 1, 'Cloyd Rivers': 5, 'Trump University': 1, 'Newsmax': 1, 'The Young Turks': 1, 'Rob Reiner': 2, 'Vox': 1, 'Ben Shapiro': 2, 'John Dingell': 1, 'U.S. News Opinion': 1, 'Ben White': 2, 'The Democrats': 1, 'Unfiltered☢Boss': 2, 'Kurt Schlichter': 1, 'Matt Viser': 1, 'Thomas': 1, 'Lara Trump': 1, 'jesse Williams.': 1, 'Judd Legum': 1, 'Michael Moore': 2, 'Aღanda': 1, 'LOLGOP': 1, 'Mark Elliott': 1, 'Charlie Kirk': 3, 'Phil 🇺🇸 Ultra MAGA': 1, 'David A. Clarke, Jr.': 2, 'John Lewis': 2, 'JustJanis': 1, 'Chase Mitchell': 1, 'Christmas Movies!': 1, 'Sam Sanders': 1, 'Tomi Lahren': 2, 'Gov. Mike Huckabee': 8, 'Gunnery Sergeant Jessie Jane Duff': 4, 'Piers Morgan': 1, 'mustard': 1, 'Thomas Paine': 3, 'Donna Brazile': 1, 'Matthew Gertz': 2, 'Matt McDermott': 1, 'John Harwood': 2, 'Gone, gone the form of man. Rise the demon Etrigan': 1, 'Michael Tracey': 1, 'Jerry Springer': 1, 'Angelo Carusone': 1, 'Eric Boehlert': 1, 'Political Polls': 1, 'billy eichner': 2, 'The General 💥': 1, 'Chad Livengood': 1, 'Carla Hayden': 1, 'Paul Ryan': 1, 'The Briefing': 6, 'Paula Reid': 1, 'Dan Perrault and Tony Yacenda are Funny or Die': 1, 'Lisa': 1, 'CNN Politics': 1, 'FOX Business': 1, 'Misha Collins': 1, 'Erin': 1, 'Sara Jacobs': 1, 'Henry J. Gomez': 1, 'Zdenek Gazda': 1, '4444444': 1, 'RSBN 🇺🇸': 1, 'Denver Post Opinion': 1, 'Al Gore': 1, 'Mic': 1, 'patback': 1, 'Dennis Mersereau': 1, 'Dan Pfeiffer': 1, 'FiveThirtyEight': 1, '🇱🇾🇬🇧جميلةJamilla': 1, 'Ben Sasse': 1, 'citizen uprising': 1, 'Zeke Miller': 1, 'The Rude Pundit': 1, 'Fox News Research': 1, 'RickLeventhal': 3, 'NadelParis': 4, 'Ari Fleischer': 1, 'PRELUDE': 1, 'Max Boot 🇺🇦': 1, 'USA TODAY': 2, 'Vogue Magazine': 1, 'NowThis': 1, 'SLOPPS': 1, 'Doug Mills': 2, 'Jonathan Chait': 1, 'Glenn Thrush': 1, 'Rebecca Shabad': 1, 'Jennifer Palmieri': 1, 'FoxNewsInsider': 1, 'Muckmaker™': 1, 'Andrew H. Scott': 1, 'Righteous⚡️Crusader': 1, 'Evan McMullin 🇺🇸': 1, 'Derek Hunter': 1, 'Asma Khalid': 1, 'picassokat': 1, "America's Voice": 1, 'Stephen Hayes': 1, 'Atif Mian': 1, 'A.J. Delgado': 2, 'Good Morning America': 1, 'Trump Super Elite': 3, 'Marlow Stern': 2, 'jonny sun': 1, 'Reuters Politics': 1, 'Jesse Tyler Ferguson (he/him/his)': 1, 'POLITICO': 2, 'eric curtin': 1, 'proto': 1, 'bettemidler': 1, 'IGNITE National': 1, 'John Fugelsang': 1, 'Lady Gaga': 1, 'Kevin M. Kruse': 1, 'Kayleigh McEnany': 2, 'Rachel Held Evans (1981-2019)': 1, 'Nick Merrill': 1, 'Elizabeth Warren': 1, 'Department of Defense 🇺🇸': 1, 'New Day': 1, 'Dr.Darrell Scott': 1, 'Ana Navarro-Cárdenas': 2, 'Daily Caller': 1, 'petesouza (archived)': 1, 'Hillary for Iowa': 1, 'Jimmy Traina': 1, 'Reince Priebus': 2, 'Greg Sargent': 1, 'Rez🏁': 1, 'John Cardillo': 2, 'Matthew Miller': 1, 'Twitter Moments': 1, 'colonel rob fee': 1, 'Alec Ross': 1, 'LA Turtle': 1, 'Kim Kardashian': 1, 'Joan Walsh': 1, 'President Applesauce Brains': 1, "Buster's Videos": 1, 'Matt Bellassai': 1, 'Singer': 1, 'Paul McLaughlin': 1, 'CNBC Opinion': 1, 'deray': 2, 'Smithsonian NMAAHC': 1, 'Ⓜ️eidas_Damon 🗽 🩸🦷': 1, 'BuzzFeed is a Chris Evans stan account': 1, 'Steve Weinstein': 1, 'Kit Daniels': 1, 'Ben Moser': 1, 'Jon Voight': 1, 'Austin Dixon': 1, 'chandler riggs': 1, 'Jeff McDevitt': 1, 'Bill Clinton': 1, 'Jon Ralston': 1, 'Bill Gates': 1, 'Angelo Ray Gomez': 1, 'Greg Greene': 1, '#ThePersistence': 1, 'Brian Deese NARA': 1, 'Paul Manafort': 1, 'Amy Mek': 1, 'Elle Mills': 1}

count_tweets = 0 
users_list = []

for key, value in unique_users.items():
    
    # print(key, ' : ', value)
    count_tweets += value
    users_list.append(key) 
   
# print(count_tweets)
# print(len(unique_users))

# Part - 2
data = pd.read_csv('Data/' + DATASET + '/New_Input.txt', sep = '<\|\|>', header=None, engine = 'python')
data.columns = ['user_id', 'user_name', 'tweet_id', 'type', 'text']
print(data.shape)

tweet_dict = {} 

for user in users_list :
    l = data[data['user_name'] == user]['text'].to_list()
    tweet_dict[user] = l

# print(tweet_dict)

# Part - 3
def CROSSEM_B(algo_tweet_list, author_tweet_list) :

    count = 0
    for author_tweet in author_tweet_list :
      
        if author_tweet in algo_tweet_list :
            count += 1
    
    return 1 if(count >= 1) else 0


def CROSSEM_F(algo_tweet_list, author_tweet_list) :

    count = 0
    for author_tweet in author_tweet_list :
      
        if author_tweet in algo_tweet_list :
            count += 1
    
    return count / len(author_tweet_list)

def CROSSEM_S(algo_tweet_list, author_tweet_list) :

    count = 0
    for author_tweet in author_tweet_list :
        for algo_tweet in algo_tweet_list : 

            data = [author_tweet, algo_tweet] 

            count_vectorizer = CountVectorizer()
            vector_matrix = count_vectorizer.fit_transform(data)

            cosine_similarity_matrix = cosine_similarity(vector_matrix)
            count += cosine_similarity_matrix[0][1]   

    return count / (len(algo_tweet_list) * len(author_tweet_list))

algo_summ_list = ['RNN-CNN']

for asl in algo_summ_list :

    binary_count = 0 
    fractional_count = 0
    similarity_count = 0

    print("Algo Summary " + asl)

    algo_summ_file = open('Data/' + DATASET + '/' + asl + '.txt', 'r')
    algo_summ = algo_summ_file.read().split("\n")

    for author in users_list : 

        author_tweets = tweet_dict[author]

        binary_count += CROSSEM_B(algo_summ, author_tweets)
        fractional_count += CROSSEM_F(algo_summ, author_tweets)
        similarity_count += CROSSEM_S(algo_summ, author_tweets)
        

    print("Binary : " , binary_count / len(tweet_dict))
    print("Fractional : " , fractional_count / len(tweet_dict))
    print("Similarity : " , similarity_count / len(tweet_dict))
    print("\n")