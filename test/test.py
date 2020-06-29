### Libraries used in making of the web application
# Streamlit dependencies
import streamlit as st
import joblib, os

# Data dependencies
import pandas as pd
from typing import List, Tuple

# Inspecting
import numpy as np
import pandas as pd
from time import time
import re
import string
import os
import emoji
from pprint import pprint
import collections

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image
import plotly.express as px
import altair as alt
from gensim.models import Word2Vec

# Balance data
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

#Natural Language Toolkit
import nltk
nltk.download('punkt')
from nltk import word_tokenize,sent_tokenize
from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
import spacy
import en_core_web_sm
sp = en_core_web_sm.load()

### Loading the data
# Vectorizer
#news_vectorizer = open("resources/tfidfvect.pkl","rb")
#tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
#raw = pd.read_csv("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/resources/train.csv")
#data2 = pd.read_csv("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/resources/test.csv")

### The main function where we will build the actual app
def main():
    ## Creating sidebar with selection box
    # Creating load data
    st.sidebar.subheader(":heavy_check_mark: Data is loaded")
    st.sidebar.text_input("link to train data", "https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/resources/train.csv")
    st.sidebar.text_input("link to test data", "https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/resources/test.csv")

	# Creating multiple pages
    st.sidebar.title("Menu")
    options = ["Homepage", "Overview of Climate Change", "Our Mission", "Machine Learning", "Data Exploration", "Models", "Our Products and Services", "About Us"]
    selection = st.sidebar.radio("Please select a page", options)

    ## Building our pages
    # Homepage page
    if selection == "Homepage":
        st.image("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/test/resources/homepage.jpg",
                  use_column_width= True)

        # Creating about us box on the side sidebar
        st.sidebar.title("About")
        st.sidebar.info("""
                    This web application is maintained by Three Musketeers. You can learn more about us on the **About Us** tab.
                    """)
        st.sidebar.image("https://i.pinimg.com/originals/5c/99/6a/5c996a625282d852811eac0ee9e81fbe.jpg",
                          use_column_width= True)


    # Overview page
    if selection == "Overview of Climate Change":

        # Added a main heading to the page
        st.image("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/test/resources/overview2%20(2).png",
                  use_column_width= True)

        st.sidebar.title("Table of contents")
        st.sidebar.info("The table of contents is interactive")
        class Toc:

            def __init__(self):
                self._items = []
                self._placeholder = None

            def title(self, text):
                self._markdown(text, "h1")

            def header(self, text):
                self._markdown(text, "h2", " " * 2)

            def subheader(self, text):
                self._markdown(text, "h3", " " * 4)

            def placeholder(self, sidebar = False ):
                self._placeholder = st.sidebar

            def generate(self):
                if self._placeholder:
                    self._placeholder.markdown("\n".join(self._items), unsafe_allow_html=True)

            def _markdown(self, text, level, space=""):
                key = "".join(filter(str.isalnum, text)).lower()

                st.markdown(f"<{level} id='{key}'>{text}</{level}>", unsafe_allow_html=True)
                self._items.append(f"{space}* <a href='#{key}'>{text}</a>")


        toc = Toc()

        toc.placeholder()

        # Video briefly explaining climate change
        st.video("https://www.youtube.com/watch?v=QG9ZcsL4lNc")

        # Brief overview of climate change
        toc.title("Climate Change")
        toc.header("❄ Weather vs. Climate ❄")
        st.image("https://climatekids.nasa.gov/review/climate-change-meaning/weather-vs-climate.png",
                  use_column_width= True)
        st.markdown("""
                    **Weather** describes the conditions happening outside right now in a specific area.

                    > _For example:_

                            If you see that it's raining outside right now, that's a way to describe today's
                            weather.

                            Rain, snow, wind, hurricanes, tornadoes — these are all weather events.

                    **Climate**, on the other hand, is more than just one or two rainy days. Climate describes the weather conditions that are expected in a region at a particular
                    time of year. Is it usually rainy or usually dry? Is it typically hot or typically cold? A region's climate is determined by observing its weather over a
                    period of many years—generally 30 years or more.

                    > _For example:_

                            One or two weeks of rainy weather wouldn't change the fact that the Karoo
                            typically has a dry, desert climate.

                            Even though it's rainy right now,
                            we still expect the Karoo to be dry because that's what is usually the case.
                    """)

        toc.header("💡 What is climate change? 💡")
        st.markdown("""
                    A significant, long-term changes in global or regional climate patterns, in particular a change apparent from the mid to late 20th century
                    onwards and attributed largely to the increased levels of atmospheric carbon dioxide produced by the use of fossil fuels.

                    The global climate is the connected system of sun, earth and oceans, wind, rain and snow, forests, deserts and savannas, and everything that
                    people do. The climate of a place, for example, Johannesburg can be described as it's rainfall, changing temperatures during the year and so on.

                    But the global climate is more than the "average" of the climated of specific places.
                    """)

        toc.subheader("The Earth has been getting warmer")
        st.image("https://ichef.bbci.co.uk/news/624/cpsprodpb/C7C4/production/_110504115_global_temperature_v2-nc.png",
                 use_column_width= True)
        st.markdown("""
                    This is linked to the greenhouse effect, which describes how the Earth's atmosphere traps some of the Sun's energy.
                    Solar energy radiating back to space from the Earth's surface is absorbed by greenhouse gases and re-emitted in all directions.

                    This heats both the lower atmosphere and the surface of the planet. Without this effect, the Earth would be about 30C colder and hostile to life.
                    """)

        toc.subheader("The greenhouse effect")
        st.image("https://climatekids.nasa.gov/review/climate-change-meaning/greenhouse-effect.gif",
                  use_column_width= True)
        st.markdown("""
                    Scientists believe we are adding to the natural greenhouse effect, with gases released from industry and agriculture trapping more energy and
                    increasing the temperature.

                    This is known as climate change or global warming.

                    **What are greenhouse gases?**

                    The greenhouse gas with the greatest impact on warming is water vapour. But it remains in the atmosphere for only a few days.

                    Carbon dioxide(CO2), however, persists for much longer. It would take hundreds of years for a return to pre-industrial levels and only so much
                    can be soaked up by natural reservoirs such as the oceans.

                    Most man-made emissions of CO2 come from burning fossil fuels. When carbon-absorbing forests are cut down and left to rot, or burned, that stored
                    carbon is released, contributing to global warming.
                    """)

        st.markdown("""

                    **_The terms "climate change" and "global warming" are used interchangeably but they mean separate things._**

                    """)
        st.image("https://snowbrains.com/wp-content/uploads/2019/08/climate_change_buzzwords.jpg?w=640",
                  use_column_width= True)

        toc.header("🌡️ What is global warming? 🌡️")
        st.markdown("""
                    Global warming is the slow increase in the average temperature of the earth's atmosphere because of an increased amount of energy(heat) striking the
                    Earth from the sun is being trapped in the atmosphere and not radiated out into space.

                    The earth's atmosphere has always acted like a greenhouse to capture the sun's heat, ensuring that the earth has enjoyed temperatures that permitted the
                    emergence of life forms as we know them. Without our atmospheric greenhouse the Earth would be very cold.

                    Today, we have the opposite problem. The problem is not that too little sun warmth is reaching the earth, but that too much is being trapped in our atmosphere.
                    So much heat is being kept inside greenhouse earth that the temperature of the earth is going up faster than at any previous time in history.
                    """)
        st.image("https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_698/https://warmheartworldwide.org/wp-content/uploads/2018/07/greenhouse-effect-image.png",
                 use_column_width = True)

        toc.subheader("How does global warming drive climate change?")
        st.markdown("""
                    All systems in the global climate system are connected, adding heat energy causes the global climate as a whole to change. Two thirds of the world is covered
                    with ocean which heats up.

                    When the ocean heats up, more water evaporates into clouds. Where storms like hurricanes and typhoons are forming, the result is
                    more energy-intensive storms. A warmer atmosphere makes glaciers and mountain snow packs, the Polar ice caps, and the great ice shield jutting off of Antarctica
                    melt which cause sea levels to rise.

                    Changes in temperature change the great patterns of wind that bring the monsoons in Asia and rain and snow around the world, making drought and unpredictable
                    weather more common.

                    This is why scientists have stopped focusing just on global warming and now focus on the larger topic of climate change.
                    """)
        st.image("https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_510,h_411/https://warmheartworldwide.org/wp-content/uploads/2018/07/Climate-change-connections.gif",
                  use_column_width = True)

        toc.header("🔥 What are the impacts of Climate change? 🔥")
        st.markdown("""
                    **Rising Sea Levels**

                    Climate change impacts rising sea levels. Average sea level around the world rose about 20 cm in the past 100 years; climate scientists expect it to
                    rise more and more rapidly in the next 100 years as part of climate change impacts.

                    Sea rise is expected entirely to submerge a number of small, island countries, and to flood coastal spawning grounds for many staple marine resources, as well
                    as low-lying capital cities, commercial agriculture, transportation and power generation infrastructure and tourism investments.
                    """)

        st.image("https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_400/https://warmheartworldwide.org/wp-content/uploads/2018/07/Rising-sea-levels.png",
                  use_column_width = True)

        st.markdown("""
                    **Melting Ice**

                    Projections suggest climate change impacts within the next 100 years, if not sooner, the world's glaciers will have disappeared, as will the Polar ice caps, and
                    the huge Antarctic ice shelf, Greenland may be green again, and snow will have become a rare phenomenon.
                    """)
        st.image("https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_300/https://warmheartworldwide.org/wp-content/uploads/2018/07/CS_arctic-sea-ice-loss_V2-300x212.png",
                 use_column_width = True)

        st.markdown("""
                    **Torrential Downpours and more powerful storms**

                    While the specific conditions that produce rainfall will not change, climate change impacts the amount of water in the atmosphere and will increase producing
                    violent downpours instead of steady showers when it does rain.
                    Hurricanes and typhoons will increase in power, and flooding will become more common.

                    Torrential downpours and devastating storms will increase large-scale damage to fields, homes, businesses, transportation and power systems and industry in
                    countries without the financial or human capital resources to respond.
                    """)
        st.image("https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_400/https://warmheartworldwide.org/wp-content/uploads/2018/07/Effects-of-climate-change.jpg",
                 use_column_width = True)
        st.image("https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_400/https://warmheartworldwide.org/wp-content/uploads/2018/07/Flooding-1.jpg",
                 use_column_width = True)

        st.markdown("""
                    **Heatwaves and droughts**

                    Despite downpours in some places, droughts and prolonged heatwaves will become common. Rising temperatures are hardly surprising, although they do not mean that
                    some parts of the world will not "enjoy" record cold temperatures and terrible winter storms.
                    (Heating disturbs the entire global weather system and can shift cold upper air currents as well as hot dry ones.)

                    Increasingly, however, hot, dry places will get hotter and drier, and places that were once temperate and had regular rainfall will become much hotter and much
                    drier.

                    Heatwaves and droughts will increase pressure on already fragile power, healthcare, water and sewage systems, as well as reducing countries' ability to feed
                    themselves or export agricultural products. Heat will also become an increasingly important killer, especially of the very young and the old.
                    """)
        st.image("https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_465,h_261/https://warmheartworldwide.org/wp-content/uploads/2018/07/EPA-project-droughts-to-end-of-century-1.gif",
                  use_column_width = True)

        st.markdown("""
                    **Changing eco-systems**

                    As the world warms, entire ecosystems will move.
                    Already rising temperatures at the equator have pushed such staple crops as rice north into once cooler areas, many fish species have migrated long distances to
                    stay in waters that are the proper temperature for them.

                    In once colder waters, this may increase fishermen's catches; in warmer waters, it may eliminate fishing; in many places, it will require fishermen to go further
                    to reach fishing grounds.

                    Farmers in temperate zones are finding drier conditions difficult for crops such as corn and wheat, and once prime growing zones are now threatened.
                    Some areas may see complete ecological change.

                    Changing ecosystems seem to result almost exclusively in the loss of important food species, for example of fish and staple crops, and the increase of malign
                    species such as disease vectors.
                    """)

        st.markdown("""
                    **Reduced food security**

                    One of the most striking impacts of rising temperatures is felt in global agriculture, although these impacts are felt very differently in the largely temperate
                    developed world and in the more tropical developing world.

                    Different crops grow best at quite specific temperatures and when those temperatures change, their productivity changes significantly.
                    At the same time, global population models suggest that developing world will add 3 billion people by 2050 and that developing world food producers must double
                    staple food crop production by then simply to maintain current levels of food consumption.

                    Food security, already shaky, is crumbling under rising temperatures and related climate changes. Major staple crops are declining in productivity, while unlike in
                    the developed countries, there are no new, more tropical staples to move in to take their places.

                    Rising population combined with declining productivity, increasing incidence of drought and storms is increasingly leaving developing countries vulnerable of food
                    shortfalls.
                    """)
        st.image("https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_300,h_225/https://warmheartworldwide.org/wp-content/uploads/2018/07/climate-change-and-food-security-in-southeast-asia-issues-and-policy-options-13-728.jpg",
                  use_column_width = True)
        st.image("https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_300,h_294/https://warmheartworldwide.org/wp-content/uploads/2018/07/climate-impacts-production1.png",
                 use_column_width = True)
        st.image("https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_300,h_225/https://warmheartworldwide.org/wp-content/uploads/2018/07/temperature-and-food-production-lost.jpg",
                 use_column_width = True)

        st.markdown("""
                    **Pests and diseases**

                    Rising temperatures favor agricultural pests, diseases and disease vectors.
                    Pest populations are on the rise and illnesses once found only in limited, tropical areas are now becoming endemic in much wider zones.

                    In Southeast Asia, for example, where malaria had been reduced to a wet season only disease in most areas, it is again endemic almost everywhere year around.
                    Likewise, dengue fever, once largely confined to tropical areas, has become endemic to the entire region.
                    Increased temperatures also increase the reproduction rates of microbes and insects, speeding up the rate at which they develop resistance to control measures
                    and drugs (a problem already observed with malaria in Southeast Asia).

                    Rising temperatures increase the reproduction rates of pests and so shorten the time required for insects and plant pathogens to develop resistance to control
                    regimes. Diseases, like pests, develop more rapidly in the heat and so do their insect vectors. Moreover, with climate change, the range of critical vectors –
                    mosquitos, for example, vectors for dengue, encephalitis, malaria, West Nile and Zika – all expand putting larger and larger populations at risk.

                    Ongoing ocean acidification threatens more and more small shell fish, which form the broad base of the ocean food chain. Ultimately, this will threaten the
                    entire ocean population and so the critical protein source for a third of the people on earth and a major industry.
                    """)
        st.image("https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_400/https://warmheartworldwide.org/wp-content/uploads/2018/07/malaria-and-cl-ch.jpg",
                 use_column_width = True)
        st.image("https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_400/https://warmheartworldwide.org/wp-content/uploads/2018/07/zika_graphic-400x200.jpg",
                 use_column_width = True)
        st.image("https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_392/https://warmheartworldwide.org/wp-content/uploads/2018/07/climate_change_health_impacts600w.jpg",
                 use_column_width = True)


        toc.title("Impact on the developing world")
        st.markdown("""
                    Climate change affects the entire globe; its impacts are more pronounced in the developing world than in the developed world.

                    In fact, ironically, although most of the human activity that produces climate change occurs in the developed world, many of climate changes' effects will
                    actually be beneficial in the developed world. In the short- and middle-term, for example, climate change will likely increase fish and agricultural yields
                    where populations are small and shrinking and productivity is highest.

                    Climate change's impacts in the developing world will be almost exclusively negative, often terribly so.

                    > As K. Smith tartly observed in 2008:

                            "The rich will find their world to be more expensive, inconvenient, uncomfortable,
                            disrupted and colourless; in general, more unpleasant and unpredictable, perhaps
                            greatly so. The poor will die."

                    """)
        st.image("https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_400/https://warmheartworldwide.org/wp-content/uploads/2018/07/Vulnerable_Countries_400.jpg",
                 use_column_width = True)
        st.image("https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_450/https://warmheartworldwide.org/wp-content/uploads/2018/07/Potential-vulnerability.png",
                  use_column_width = True)


        toc.header("🌍 How can they slow down climate change? 🌍")
        st.markdown("""
                    Countries in the developing world can make two major contributions to slowing climate change:
                    They can pursue smart development, avoiding the worst mistakes of the developed world; and they can reduce – even reverse – their one major contribution to
                    climate change: unsustainable agriculture practices.
                    """)
        st.subheader("What can the developing world do to avoid the mistakes of the developed world?")
        st.markdown("""
                    Look first at the primary sources of the greenhouse gasses that cause global warming: Power generation (25%); industry (21%); transportation (14%); and
                    buildings (6%).

                    **Power**

                    Most power is generated in the developed world, much using old, dirty technology and carried long distances over inefficient power grids. Developing countries
                    have the opportunity to build entirely new, distributed generation power systems that require no grids and use non-polluting technologies.

                    **Industry**

                    Building greenfield industrial economies, developing countries have the opportunity to cost the environment and construct with non-polluting technologies.

                    **Transportation**

                    Not yet entirely dependent upon massive road-based transportation infrastructures, developing countries have the opportunity to design efficient, low-cost,
                    high volume transportation systems to serve cities and industrial centers, and to use policy incentives to discourage personal automobile ownership and construct
                    high quality public transportation systems.

                    **Building**
                    And because so much existing building stock must be replaced in short order, developing countries have the opportunity to build efficiency into individual structures
                    and to design urban areas for high density, high energy efficiency living.

                    Excellent models already exist in China, Korea and Singapore, and even the medium-term cost savings are so great that not investing to do better than the developed
                    world today is foolish.
                    """)
        st.video("https://youtu.be/G4H1N_yXBiA")

        toc.generate()


    # Our Mission page
    if selection == "Our Mission":
        st.image("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/test/resources/ourmission.png",
                  use_column_width= True)

        st.sidebar.title("Table of contents")
        st.sidebar.info("The table of contents is interactive")
        class Toc1:
            def __init__(self):
                self._items = []
                self._placeholder = None

            def title(self, text):
                self._markdown(text, "h1")

            def header(self, text):
                self._markdown(text, "h2", " " * 2)

            def subheader(self, text):
                self._markdown(text, "h3", " " * 4)

            def placeholder(self, sidebar = False ):
                self._placeholder = st.sidebar

            def generate(self):
                if self._placeholder:
                    self._placeholder.markdown("\n".join(self._items), unsafe_allow_html=True)

            def _markdown(self, text, level, space=""):
                key = "".join(filter(str.isalnum, text)).lower()

                st.markdown(f"<{level} id='{key}'>{text}</{level}>", unsafe_allow_html=True)
                self._items.append(f"{space}* <a href='#{key}'>{text}</a>")


        toc1 = Toc1()

        toc1.placeholder()

        st.image("https://media.coveringmedia.com/media/images/movies/2011/10/16/three_title.jpg",
                 use_column_width= True)

        toc1.title("⚔️ The Three Musketeers ⚔️")
        st.markdown("""
                    Our goal is to combine Machine Learning techniques and user-frinedly interfaces and technology to educate  people and businesses around climated
                    change.

                    We use Machine Learning models to classify a persons comments about climate change whether they are for or against climate change.

                    By bridging the gap between Machine Learning and human understanding and able to deliver actionable insights and information that can allow or inform
                    future strategies around climate change.
                    """)

        toc1.header("Synopsis")
        st.markdown("""
                    Many companies are built around lessening one's environmental impact or carbon footprint. They offer products and services that are environmentally friendly
                    and sustainable, in line with their values and ideals. They would like to determine how people perceive climate change and whether or not they believe it is a
                    real threat. This would add to their market research efforts in gauging how their product or service may be received.

                    Creating a Machine Learning model that is able to classify whether or not a person believes in climate change, based on their novel tweet data.

                    Providing an accurate and robust solution to this task gives companies access to a broad base of consumer sentiment, spanning multiple demographic and geographic
                    categories - thus increasing their insights and informing future marketing strategies.
                    """)

        toc1.header("Our Approach")
        st.markdown("""
                    **Summary of analysis before modelling**

                    The exploratory data analysis gave insight into which words were being used the most, in terms of a politcal figure and famous people it was Donald Trump, Obama,
                    Leonardo Dicaprio, Scott Pruis, Al Gore, Steves Goddard. Organisations that appeared a lot in these tweets were EPA, Exxon and UN. Another discovery was that appeared
                    a lot in the tweets had 'RT' which means that people retweeted their feelings of sentiment instead of saying anything personal. The countries that appeared the most
                    were USA, China and the city Paris.

                    The words that appeared the most from the WordClouds are fight climate, scientist, news, warm urlweb (urlweb means a link was originally here), change denier,
                    not believe, believe climate, cause global and the realdonaldtrump.

                    The people that are metioned above play a major role in climate change. Trump's first term has been a relentless drive for no restrictions on fossil energy development.
                    This in other words was to create more jobs and GDP for America with no sentiment towards the environment. He changed all the laws that Obama started to protect the
                    environment such as the safety rules for for offshore drilling operations.

                    Leonardo Dicaprio has played a monumental movement in helping climate change where he has a foundation called the Leonardo DiCaprio Foundation, this is intuitively why
                    his name appears a lot in the tweets. Steve Goddard is an environmentalist and is on both sides of the climate debate. He is famous for writing the article The Register,
                    which describes Arctic Sea ice is not receding and claimed that data from the National Snow and Ice Data Center (NSIDC) showing the opposite was incorrect. China is the
                    worlds largest emiiter of carbon dioxide since the economy is growing on a large scale and they have one of the hugest populations.

                    The second country that emmits the most carbon dioxide is the USA. The Paris climate change agreement is the reasons for the city to appear so much in the tweets. The agreement
                    includes commitments from all major emitting countries to cut their climate-altering pollution and to strengthen those commitments over time. This is some of the insight found
                    from analysis and you can see why these organisations, places, people and choice of words have appeared on the topic of climate change.

                    The preprocessing of data was done based on the statistical analysis of counting each of the specific characters to see if it added to sentiment.

                    > _The following was noticed:_

                            - The increase in url links showed a sentiment of 2: thats why the link was
                              replaced with 'urlweb'.

                            - Punctuation such as question and exclamation marks added no sentiment.

                            - The word count did not favour any sentiment therefore not added as a parameter.

                            - The emojicons added no value to sentiment so it was replaced with 'emoji' in
                              order to not lose information.

                            - The capatalised words were few and did not add to sentiment therefore the tweets
                              were made as all lower case. The only aspects left of the tweet were lowercase
                              words and no other symbols.

                            - The stopwords words used a process called stemming to get the routes of the
                              words, this would help the process of vectorization. The stopwords were removed
                              to decrease the noise from the the actual words that add sentiment.

                            - The repeated rows in the train set were removed as it would not add any insight
                              in the training the model.

                    """)

        toc1.generate()

    # Machine Learning Page
    if selection == "Machine Learning":
        st.image("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/test/resources/MachineLearning.png",
                  use_column_width= True)

        st.sidebar.title("Table of contents")
        st.sidebar.info("The table of contents is interactive")
        class Toc2:
            def __init__(self):
                self._items = []
                self._placeholder = None

            def title(self, text):
                self._markdown(text, "h1")

            def header(self, text):
                self._markdown(text, "h2", " " * 2)

            def subheader(self, text):
                self._markdown(text, "h3", " " * 4)

            def placeholder(self, sidebar = False ):
                self._placeholder = st.sidebar

            def generate(self):
                if self._placeholder:
                    self._placeholder.markdown("\n".join(self._items), unsafe_allow_html=True)

            def _markdown(self, text, level, space=""):
                key = "".join(filter(str.isalnum, text)).lower()

                st.markdown(f"<{level} id='{key}'>{text}</{level}>", unsafe_allow_html=True)
                self._items.append(f"{space}* <a href='#{key}'>{text}</a>")


        toc2 = Toc2()

        toc2.placeholder()

        st.image("https://expertsystem.com/wp-content/uploads/2017/03/machine-learning-definition.jpeg",
                 use_column_width= True)

        toc2.title("What is Machine Learning?")
        st.markdown("""
                    Machine learning is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without
                    being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it learn for themselves.

                    The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make
                    better decisions in the future based on the examples that we provide. The primary aim is to allow the computers learn automatically without human intervention
                    or assistance and adjust actions accordingly.

                    But, using the classic algorithms of machine learning, text is considered as a sequence of keywords; instead, an approach based on semantic analysis mimics the
                    human ability to understand the meaning of a text.
                    """)


        toc2.header("🤖 Machine Learning Methods 🤖")
        st.markdown("""
                    Machine learning algorithms are often categorized as supervised or unsupervised.
                    """)

        st.image("https://blog.bismart.com/hs-fs/hubfs/Machinne%20Learning%20Types%20Bismart.png?width=900&name=Machinne%20Learning%20Types%20Bismart.png",
                 use_column_width= True)

        toc2.subheader("Supervised Machine Learning techniques")
        st.markdown("""
                    Supervised machine learning algorithms can apply what has been learned in the past to new data using labeled examples to predict future events. Starting from
                    the analysis of a known training dataset, the learning algorithm produces an inferred function to make predictions about the output values. The system is able
                    to provide targets for any new input after sufficient training.

                    The learning algorithm can also compare its output with the correct, intended output and find
                    errors in order to modify the model accordingly.
                    """)
        st.image("https://miro.medium.com/max/929/1*U5GzWYAm8t1urqdDxpiOFw.jpeg",
                 use_column_width= True)

        toc2.subheader("Unsupervised Machine Learning techniques")
        st.markdown("""
                    Unsupervised machine learning algorithms are used when the information used to train is neither classified nor labeled. Unsupervised learning studies how systems
                    can infer a function to describe a hidden structure from unlabeled data. The system doesn't figure out the right output, but it explores the data and can draw
                    inferences from datasets to describe hidden structures from unlabeled data.
                    """)
        st.image("https://www.newtechdojo.com/wp-content/uploads/2018/03/How-unsupervised-machine-Learning-works.jpg",
                 use_column_width= True)


        toc2.title("Natural Language Processing?")
        st.image("https://opendatascience.com/wp-content/uploads/2020/05/Image-for-ODSC-article-750x350.jpg",
                 use_column_width= True)
        st.markdown("""
                    Natural Language Processing (NLP) is the technology used to aid computers to understand the human's natural language. It's not an easy task teaching
                    machines to understand how we communicate.
                    """)

        toc2.header("߷ What is Natural Language Processing? ߷")
        st.markdown("""
                    Natural Language Processing is a branch of artificial intelligence (AI) that deals with the interaction between computers and humans using the
                    natural language. The ultimate objective of NLP is to read, decipher, understand, and make sense of the human languages in a manner that is valuable.
                    Most NLP techniques rely on machine learning to derive meaning from human languages.

                    > _Typical interaction between humans and machines using Natural Language Processing
                      could go as follows:_

                                1. A human talks to the machine
                                2. The machine captures the audio
                                3. Audio to text conversion takes place
                                4. Processing of the text's data
                                5. Data to audio conversion takes place
                                6. The machine responds to the human by playing the audio file
                    """)
        st.image("https://cdn.business2community.com/wp-content/uploads/2019/04/NLP_Infog-01-900x427.png",
                 use_column_width= True)

        toc2.header("🔗 What is NLP used for? 🔗")
        st.markdown("""
                    > _Natural Language Processing is the driving force behind the following common applications:_

                            - Language translation applications such as Google Translate
                            - Word Processors such as Microsoft Word and Grammarly that employ NLP to check
                              grammatical accuracy of texts.
                            - Interactive Voice Response (IVR) applications used in call centers to respond to
                              certain users' requests.
                            - Personal assistant applications such as OK Google, Siri, Cortana, and Alexa.
                    """)

        toc2.header("🖥️ Why is NLP difficult? 🖥️")
        st.markdown("""

                    Natural Language processing is considered a difficult problem in computer science. It's the nature of the human language that makes NLP difficult.

                    The rules that dictate the passing of information using natural languages are not easy for computers to understand.

                    Some of these rules can be high-leveled and abstract:
                    > _for example:_

                                    When someone uses a sarcastic remark to pass information.

                    On the other hand, some of these rules can be low-levelled:
                    > _for example:_

                                    Using the character "s" to signify the plurality of items.


                    Comprehensively understanding the human language requires understanding both the words and how the concepts are connected to deliver the intended message.

                    While humans can easily master a language, the ambiguity and imprecise characteristics of the natural languages are what make NLP difficult for machines to implement.
                    """)

        toc2.header("💾 How does NLP Works? 💾")
        st.markdown("""
                    NLP entails applying algorithms to identify and extract the natural language rules such that the unstructured language data is converted into a form that
                    computers can understand.

                    When the text has been provided, the computer will utilize algorithms to extract meaning associated with every sentence and collect the essential data from them.
                    Sometimes, the computer may fail to understand the meaning of a sentence well, leading to obscure results.

                    > _for example:_

                      A funny incident occurred in the 1950s during the translation of some words between
                      the English and the Russian languages.

                            Here is the biblical sentence that required translation:
                                "The spirit is willing, but the flesh is weak."

                            Here is the result when the sentence was translated to Russian and back to
                            English:
                                "The vodka is good, but the meat is rotten."

                    """)

        toc2.header("⌨️ What are the techniques used in NLP? ⌨️")
        st.markdown("""
                    Syntactic analysis and semantic analysis are the main techniques used to complete Natural Language Processing tasks.
                    """)

        toc2.subheader("Syntax")
        st.markdown("""
                    Syntax refers to the arrangement of words in a sentence such that they make grammatical sense.

                    In NLP, syntactic analysis is used to assess how the natural language aligns with the grammatical rules.
                    Computer algorithms are used to apply grammatical rules to a group of words and derive meaning from them.

                    > _Here are some syntax techniques that can be used:_

                    **Lemmatization:** It entails reducing the various inflected forms of a word into a single form for easy analysis.""")
        st.image("https://cdn-images-1.medium.com/max/1600/1*z4f7My5peI28lNpZdHk_Iw.png",
                 use_column_width = True)

        st.markdown("""**Morphological segmentation:** It involves dividing words into individual units called morphemes.""")

        st.image("https://images.slideplayer.com/2/685592/slides/slide_16.jpg",
                 use_column_width = True)

        st.markdown("""**Word segmentation:** It involves dividing a large piece of continuous text into distinct units.""")

        st.image("https://miro.medium.com/max/813/1*NOERSs9amkoYnCHQE5s3DA.png",
                 use_column_width = True)

        st.markdown("""**Part-of-speech tagging:** It involves identifying the part of speech for every word.""")

        st.image("https://slideplayer.com/slide/5260592/16/images/2/What+is+POS+tagging+Tagged+Text+Raw+Text+POS+Tagger.jpg",
                 use_column_width = True)
        st.image("https://i.imgur.com/lsmcqqk.jpg",
                 use_column_width = True)

        st.markdown("""**Parsing:** It involves undertaking grammatical analysis for the provided sentence.""")
        st.image("https://www.bowdoin.edu/~allen/nlp/fig1.GIF", use_column_width = True)

        st.markdown("""**Sentence breaking:** It involves placing sentence boundaries on a large piece of text.""")
        st.image("https://image.slidesharecdn.com/nlppipelineinmachinetranslation-170307231905/95/nlp-pipeline-in-machine-translation-9-638.jpg?cb=1488928901",
                  use_column_width = True)

        st.markdown("""**Stemming:** It involves cutting the inflected words to their root form.""")
        st.image("https://cdn-images-1.medium.com/max/1600/1*9KLZmtSh-t6SfEGX_nbznA.png", use_column_width = True)

        toc2.subheader("Semantics")
        st.markdown("""
                    Semantics refers to the meaning that is conveyed by a text. Semantic analysis is one of the difficult aspects of Natural Language Processing that
                    has not been fully resolved yet.

                    It involves applying computer algorithms to understand the meaning and interpretation of words and how sentences are structured.
                    > _Here are some techniques in semantic analysis:_

                          - Named entity recognition (NER): It involves determining the parts of a text that
                                                            can be identified and categorized into preset
                                                            groups.

                                                            Examples of such groups include names of people
                                                            and names of places.

                          - Word sense disambiguation: It involves giving meaning to a word based on the
                                                       context.

                          - Natural language generation: It involves using databases to derive semantic
                                                         intentions and convert them into human language.
                    """)
        st.image("https://storage.ning.com/topology/rest/1.0/file/get/5104454460?profile=RESIZE_710x",
                  use_column_width = True)

        toc2.title("Supervised Machine Learning")
        st.image("https://miro.medium.com/max/1400/1*t8igzDSUC-1qaJQRQpbwBA.png",
                 use_column_width = True)
        toc2.header("📚 Classification 📚")
        toc2.subheader("What is classification?")
        st.markdown("""
                    Classification is a process of categorizing a given set of data into classes, It can be performed on both structured or unstructured data. The process starts with
                    predicting the class of given data points. The classes are often referred to as target, label or categories.

                    The classification predictive modeling is the task of approximating the mapping function from input variables to discrete output variables. The main goal is to
                    identify which class/category the new data will fall into.
                    """)
        st.image("https://d1jnx9ba8s6j9r.cloudfront.net/blog/wp-content/uploads/2019/11/classification.png",
                 use_column_width = True)

        toc2.subheader("Terminology")
        st.markdown("""
                     - **Classifier:** It is an algorithm that is used to map the input data to a specific category.

                     - **Classification Model:** The model predicts or draws a conclusion to the input data given for training, it will predict the class or category for the data.

                     - **Feature:** A feature is an individual measurable property of the phenomenon being observed.

                     - **Binary:** Classification – It is a type of classification with two outcomes, for eg – either true or false.

                     - **Multi-Class Classification:** The classification with more than two classes, in multi-class classification each sample is assigned to one and only one label or target.

                     - **Multi-label Classification:** This is a type of classification where each sample is assigned to a set of labels or targets.

                     - **Initialize:** It is to assign the classifier to be used for the

                     - **Train the Classifier:** Each classifier in sci-kit learn uses the fit(X, y) method to fit the model for training the train X and train label y.

                     - **Predict the Target:** For an unlabeled observation X, the predict(X) method returns predicted label y.

                     - **Evaluate:** This basically means the evaluation of the model i.e classification report, accuracy score, etc.
                    """)

        toc2.subheader("Models")
        st.markdown("""
                    In machine learning, classification is a supervised learning concept which basically categorizes a set of data into classes.

                    The most common classification problems are – speech recognition, face detection, handwriting recognition, document classification, etc. It can be either a
                    binary classification problem or a multi-class problem too.

                    **Logistic Regression**

                    It is a classification algorithm in machine learning that uses one or more independent variables to determine an outcome. The outcome is measured with a
                    dichotomous variable meaning it will have only two possible outcomes.

                    The goal of logistic regression is to find a best-fitting relationship between the dependent variable and a set of independent variables. It is better than
                    other binary classification algorithms like nearest neighbor since it quantitatively explains the factors leading to classification.

                    > _Advantages and Disadvantages_

                            Logistic regression is specifically meant for classification, it is useful in
                            understanding how a set of independent variables affect the outcome of the
                            dependent variable.

                            The main disadvantage of the logistic regression algorithm is that it only works
                            when the predicted variable is binary, it assumes that the data is free of missing
                            values and assumes that the predictors are independent of each other.

                    > _Best Use Cases_

                            - Identifying risk factors for diseases
                            - Word classification
                            - Weather Prediction
                            - Voting Applications
                    """)

        st.image("https://d1jnx9ba8s6j9r.cloudfront.net/blog/wp-content/uploads/2019/11/log.png",
                 use_column_width = True)

        st.markdown("""
                    **Naive Bayes Classifier**

                    It is a classification algorithm based on Bayes’s theorem which gives an assumption of independence among predictors. In simple terms, a Naive Bayes classifier
                    assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.

                    Even if the features depend on each other, all of these properties contribute to the probability independently. Naive Bayes model is easy to make and is particularly
                    useful for comparatively large data sets. Even with a simplistic approach, Naive Bayes is known to outperform most of the classification methods in machine learning.
                    Following is the Bayes theorem to implement the Naive Bayes Theorem.

                    > _Advantages and Disadvantages_

                            The Naive Bayes classifier requires a small amount of training
                            data to estimate the necessary parameters to get the results.
                            They are extremely fast in nature compared to other classifiers.

                            The only disadvantage is that they are known to be a bad estimator.

                    > _Best Use Cases_

                            - Disease Predictions
                            - Document Classification
                            - Spam Filters
                            - Sentiment Analysis
                    """)

        st.image("https://d1jnx9ba8s6j9r.cloudfront.net/blog/wp-content/uploads/2019/11/naive.png",
                 use_column_width = True)

        st.markdown("""
                    **K-Nearest Neighbor**

                    It is a lazy learning algorithm that stores all instances corresponding to training data in n-dimensional space. It is a lazy learning algorithm as it does not
                    focus on constructing a general internal model, instead, it works on storing instances of training data.

                    Classification is computed from a simple majority vote of the k nearest neighbors of each point. It is supervised and takes a bunch of labeled points and uses
                    them to label other points. To label a new point, it looks at the labeled points closest to that new point also known as its nearest neighbors. It has those
                    neighbors vote, so whichever label the most of the neighbors have is the label for the new point. The “k” is the number of neighbors it checks.

                    > _Advantages And Disadvantages_

                            This algorithm is quite simple in its implementation and is robust to
                            noisy training data. Even if the training data is large, it is quite
                            efficient.

                            The only disadvantage with the KNN algorithm is that there
                            is no need to determine the value of K and computation cost is pretty
                            high compared to other algorithms.

                    > _Best Use Cases_

                        - Industrial applications to look for similar tasks in comparison to others
                        - Handwriting detection applications
                        - Image recognition
                        - Video recognition
                        - Stock analysis
                    """)
        st.image("https://d1jnx9ba8s6j9r.cloudfront.net/blog/wp-content/uploads/2019/11/knn.png",
                 use_column_width = True)

        st.markdown("""
                    The support vector machine is a classifier that represents the training data as points in space separated into categories by a gap as wide as possible.
                    New points are then added to space by predicting which category they fall into and which space they will belong to.

                    > _Advantages and Disadvantages_

                            It uses a subset of training points in the decision function which makes
                            it memory efficient and is highly effective in high dimensional spaces.

                            The only disadvantage with the support vector machine is that the
                            algorithm does not directly provide probability estimates.

                    > _Best Use cases_

                            - Business applications for comparing the performance of a stock over a
                              period of time
                            - Investment suggestions
                            - Classification of applications requiring accuracy and efficiency
                    """)
        st.image("https://d1jnx9ba8s6j9r.cloudfront.net/blog/wp-content/uploads/2019/11/svm-2-1.png",
                 use_column_width = True)

        toc2.header("📈 Regression 📈")
        toc2.subheader("Brief Overview")
        st.markdown("""
                    Regression is another form of supervised learning. The difference between classification and regression is that regression outputs a number rather than a class.
                    Therefore, regression is useful when predicting number based problems like stock market prices, the temperature for a given day, or the probability of an event.
                    """ )
        st.image("https://miro.medium.com/max/1400/1*LbvtG6tsDwTeS7QPn6NfGw.png",
                 use_column_width = True)


        toc2.generate()


    # Data Exploration
    if selection == "Data Exploration":
        st.image("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/test/resources/dataexploration.png",
                  use_column_width= True)

        # EDA
        my_dataset = "https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/resources/train.csv"

        #Lemmetization and Stemming
        st.subheader("**_Lemmetization and Stemming_**")
        st.markdown('''Type in words with a space inbetween and the two methods will return the route word''')
        st.info("Predict Lemmetization and  Stemming of your own words")
		# Creating a text box for user input
        tweet_text_ls = st.text_area("Enter Text","Type Here")

        #Lemmetization Predictor
        if st.button('Lemmetization'):
            text = sp(tweet_text_ls)
            pred_l = []
            for word in text:
                pred_l.append('Lemma for '+str(word)+' is '+str(word.lemma_))
            for p in pred_l:
                st.success("{}".format(p))

        #Stemming Predictor
        if st.button('Stemming'):
            stemmer = PorterStemmer()
            tokenizer = nltk.word_tokenize(tweet_text_ls)
            pred_l = []
            for token in tokenizer:
                pred_l.append('Stem for '+token+' is '+stemmer.stem(token))
            for p in pred_l:
                st.success("{}".format(p))

        #Info
        st.subheader("**_Original Tweets_**")
        st.info('View Original Data Set')

        # To Improve speed and cache data
        @st.cache(persist=True,allow_output_mutation=True)
        def explore_data(dataset):
            df = pd.read_csv(os.path.join(dataset))
            return df

        # Our Dataset
        data = explore_data(my_dataset)

        # Show raw Dataset
        if st.checkbox("Preview DataFrame"):

            if st.button("Head"):
                st.write(data.head())
            if st.button("Tail"):
                st.write(data.tail())
            else:
                st.write(data.head(2))

        # Show Entire Dataframe
        if st.checkbox("Show All DataFrame"):
            st.dataframe(data)

        #Define Dataframe for pie chart plot
        df_pie = data.groupby('sentiment').count().reset_index()
        df_pie['sentiment'].replace([-1,0,1,2],['negative Sentiment = -1','neutral Sentiment = 0','positve Sentiment = 1','News Sentiment = 2'],inplace =True)

        #Markdown explaining the distribtion of Target
        st.subheader("**_Distribution of Target_**")
        st.markdown('<p><ul><li>The positive sentiment counts are significantly higher followed by news, then neutral and lastly anti.', unsafe_allow_html=True)

        #Show distribution of target variable
        st.info('View Distribution of Sentiment')
        if st.button('Bar Plot'):
            @st.cache(persist=True,allow_output_mutation=True)
            def figure1(df):
                fig = sns.factorplot('sentiment',data = df, kind='count',size=6,aspect = 1.5, palette = 'PuBuGn_d')
                return fig
            fig1 = figure1(data)
            st.markdown("<h1 style='text-align: center; color: black;'>Distribution of Sentiment</h1>", unsafe_allow_html=True)
            st.pyplot(fig1)
        if st.button('Pie Chart'):
            @st.cache(persist=True,allow_output_mutation=True)
            def figure2(df):
                fig = px.pie(df, values='message', names='sentiment',color_discrete_map={'negative Sentiment = -1':'lightcyan','neutral Sentiment = 0':'cyan','positve Sentiment = 1':'royalblue','News Sentiment = 2':'darkblue'})
                return fig
            fig2 = figure2(df_pie)
            st.markdown("<h1 style='text-align: center; color: black;'>Climate Sentiment Pie Chart</h1>", unsafe_allow_html=True)
            st.plotly_chart(fig2, use_container_width=True)

        #markdown to explain the clean data
        st.subheader("**_Clean Tweets_**")
        st.markdown("""
					<p><ul><li> Firslt, the cleaning of the data followed a process of using <a href="https://docs.python.org/3/howto/regex.html" target="_blank">Regex</a> to remove capital words,
					replace urls, replace emojis, remove digits only keep certain characters within the text. For more information,
					you may look at the following link <a href="https://towardsdatascience.com/sentiment-analysis-with-text-mining-13dd2b33de27l" target="_blank">Sentiment Analysis</a></li>
					<li> Secondly, the following methods were used to enable the natural language process library built
					in python in order to clean the texts further. These methods were, <a href="https://www.nltk.org/api/nltk.tokenize.html" target="_blank">tokenization</a>,  <a href="https://pythonprogramming.net/stemming-nltk-tutorial/" target="_blank">stemming</a>
					and lastly removal of <a href="https://www.nltk.org/book/ch02.html" target="_blank">stopwords</a></li>
					<li>Finally, the cleaned tweets were transformed from a list (due to tokenization) to a string.</li></ul></p>
					""",unsafe_allow_html=True)

        #Cleaning of text before tokenisation, stemming and removal of stop words
        clean_train_df = pd.read_csv("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/resources/Clean_Train.csv")
        clean_test_df = pd.read_csv("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/resources/Clean_Test.csv")

        #Define Dataframe for more Analysis
        EDA_df = clean_train_df[['sentiment','clean_tweet']]

        #Info
        st.info('View Clean Data Set')

        #View Clean Data
        @st.cache(persist=True)
        def explore_data_clean(df):
            df1 = df
            return df1

        # Our clean Dataset
        data_clean = explore_data_clean(EDA_df)

        # Show clean Dataset
        if st.checkbox("Preview showing clean DataFrame"):

            if st.button("Head of Clean Data"):
                st.write(data_clean.head())
            if st.button("Tail of Clean Data"):
                st.write(data_clean.tail())
            else:
                st.write(data_clean.head(2))

        # Show Entire Dataframe
        if st.checkbox("Show All  of Clean Dataframe"):
            st.dataframe(data_clean)

        #Preper Word2Vec
        @st.cache(persist=True,allow_output_mutation=True)
        def token(df):
            df1 = df['clean_tweet'].apply(lambda x: x.split()) #tokenising
            return df1
        tokenised_tweet = token(clean_train_df)

        #create list of words with no repetitions
        all_words =[]
        for index, rows in clean_train_df.iterrows():
            all_words.append(rows['clean_tweet'].split(' '))
        flatlist_all = [item for sublist in all_words for item in sublist]
        single_list_of_words = list(set(flatlist_all))

        #Word2Vec
        st.subheader("**_Word2Vec of Clean Tweets_**")
        st.markdown('''It will return the probability of other words appearing from the word you typed in''')
        st.info("Type in word from tweets that can be observed above")
        # Creating a text box for user input
        tweet_text_vec = st.text_area("Enter Text","Eg: realdonaldtrump")

        #Predict similar words
        if st.button('Predict Similar Words'):
            if tweet_text_vec in single_list_of_words:
                @st.cache(persist=True)
                def word2vec(text):
                    model_w2v = Word2Vec(
                                            tokenised_tweet,
									        size=200, # desired no. of features/independent variables
									        window=5, # context window size
									        min_count=2,
									        sg = 1, # 1 for skip-gram model
									        hs = 0,
									        negative = 10, # for negative sampling
									        workers= 2, # no.of cores
									        seed = 34)
                    model_w2v.train(tokenised_tweet,total_examples= len(clean_train_df['clean_tweet']), epochs=20)
                    vec = model_w2v.wv.most_similar(positive=text)
                    return vec
                predict_vec = word2vec(tweet_text_vec)
                for tuple in predict_vec:
                    st.success("{}".format(tuple))
                else:
                    st.success('Word Not found, please try again')

        #WordCloud Creation
		#Sentiment of 2

		#Sentiment of -1

        #Markdown for WordCloud
        st.subheader('**_WordCloud Plots of Clean Tweets_**')
        st.markdown('''
					<p>Plotting a <a href="https://www.geeksforgeeks.org/generating-word-cloud-python/" target="_blank">WordCloud</a> will help the common words used in a tweet. The most important analysis is understanding
					sentiment and the wordcloud will show the common words used by looking at the train dataset</p>
					''', unsafe_allow_html=True)

        #Info
        st.info('WordClouds')

        if st.button("sentiment 2"):
            st.image('https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/resources/imgs/WordCloud2.PNG')
        if st.button("sentiment 1"):
            st.image('https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/resources/imgs/WordCloud1.PNG')
        if st.button('sentiment 0'):
            st.image('https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/resources/imgs/WordCloud0.PNG')
        if st.button('sentiment -1'):
            st.image('https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/resources/imgs/WordCloud-1.PNG')

		#Create your own wordcloud

        #info
        st.subheader("**_Create Your Own WordCloud_**")
        st.markdown('''Type in in a long paragraph and see what the WordCloud generates''')
        st.info("WordCloud Creator")

        list_test = ['hello', 'test','testing','please']
        # Creating a text box for user input
        tweet_text_cloud = st.text_area("Enter Text Here","Type Text Here")
        tweet_text_list = tweet_text_cloud.split( )
        if len(tweet_text_list) >= 1:
            list_test = tweet_text_list

        #Create WorldCloud
        words =' '.join([text for text in list_test])
        wordcloud = WordCloud(background_color ='white',width=1000, height=750, random_state=21, max_font_size=300).generate(words)

        #Show WordCloud Visually
        if st.button("Create"):
            plt.imshow(wordcloud)
            plt.axis("off")
            st.markdown("<h1 style='text-align: center; color: black;'>Your Own WordCloud</h1>", unsafe_allow_html=True)
            plt.show()
            st.pyplot()

        #Hashtags
        st.subheader('**_Hashtag Plots of Clean Tweets_**')
        st.markdown('''
					<p>The hashtags were plotted per sentiment as people use '#' in tweets
					before a relevant keyword or phrase in their tweets.
					''', unsafe_allow_html=True)

        # function to collect hashtags
        @st.cache(persist=True,allow_output_mutation=True)
        def hashtag_extract(x):
            hashtags = []
            # Loop over the words in the tweet
            for i in x:
                ht = re.findall(r"#(\w+)", i)
                hashtags.append(ht)

            return hashtags

        # extracting hashtags from  tweets
        HT_neutral = hashtag_extract(clean_train_df['clean_tweet'][clean_train_df['sentiment'] == 0])
        HT_pro = hashtag_extract(clean_train_df['clean_tweet'][clean_train_df['sentiment'] == 1])
        HT_news = hashtag_extract(clean_train_df['clean_tweet'][clean_train_df['sentiment'] == 2])
        HT_anti = hashtag_extract(clean_train_df['clean_tweet'][clean_train_df['sentiment'] == -1])

        # unnesting list
        HT_neutral = sum(HT_neutral,[])
        HT_pro = sum(HT_pro,[])
        HT_news = sum(HT_news,[])
        HT_anti = sum(HT_anti,[])

        #Plotting Hashtags
        #Info
        st.info('Hashtags')


        if st.button("Sentiment 2"):
            @st.cache(persist=True,allow_output_mutation=True)
            def hashtag1(lst):
                a = nltk.FreqDist(lst)
                d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})
                # selecting top 5 most frequent hashtags
                d = d.sort_values(by = 'Count',ascending = False)
                return d[0:5]
            hash1 = hashtag1(HT_news)
            st.markdown("<h1 style='text-align: center; color: black;'> Hashtag for News(2) Sentiment</h1>", unsafe_allow_html=True)
            sns.barplot(data=hash1, x= "Hashtag", y = "Count")
            st.pyplot()
        if st.button("Sentiment 1"):
            @st.cache(persist=True,allow_output_mutation=True)
            def hashtag2(lst):
                a = nltk.FreqDist(lst)
                d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})
                # selecting top 5 most frequent hashtags
                d = d.sort_values(by = 'Count',ascending = False)
                return d[0:5]
            hash2 = hashtag2(HT_pro)
            st.markdown("<h1 style='text-align: center; color: black;'> Hashtag for Postive(1) Sentiment</h1>", unsafe_allow_html=True)
            sns.barplot(data=hash2, x= "Hashtag", y = "Count")
            st.pyplot()

        if st.button('Sentiment 0'):
            @st.cache(persist=True,allow_output_mutation=True)
            def hashtag3(lst):
                a = nltk.FreqDist(lst)
                d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})
                # selecting top 5 most frequent hashtags
                d = d.sort_values(by = 'Count',ascending = False)
                return d[0:5]
            hash3 = hashtag3(HT_neutral)
            st.markdown("<h1 style='text-align: center; color: black;'> Hashtag for Neutral(0) Sentiment</h1>", unsafe_allow_html=True)
            sns.barplot(data=hash3, x= "Hashtag", y = "Count")
            st.pyplot()

        if st.button('Sentiment -1'):
            @st.cache(persist=True,allow_output_mutation=True)
            def hashtag4(lst):
                a = nltk.FreqDist(lst)
                d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})
                # selecting top 5 most frequent hashtags
                d = d.sort_values(by = 'Count',ascending = False)
                return d[0:5]
            hash4 = hashtag4(HT_anti)
            st.markdown("<h1 style='text-align: center; color: black;'> Hashtag for Negative(-1) Sentiment</h1>", unsafe_allow_html=True)
            sns.barplot(data=hash4, x= "Hashtag", y = "Count")
            st.pyplot()


    # Models
    if selection == "Models":
        st.image("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/test/resources/Predictivemodel.png",
                  use_column_width= True)

        st.info("Prediction with ML Models")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text","Type Here")

        if st.button("Classifier: LinearSVC"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
            prediction = joblib.load(open(os.path.join("test/resources/LinearSVC_model.pkl"),"rb"))
            prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
            if prediction == 2:
                st.success('Sentiment prediction: 2 (Very Postive Sentiment)')
            if prediction == 1:
                st.success('Sentiment prediction: 1 (Postive Sentiment)')
            if prediction == 0:
                st.success('Sentiment prediction: 0 (Neutral Sentiment)')
            if prediction == -1:
                st.success('Sentiment prediction: -1 (Negative Sentiment)')

        if st.button("Classifier: LogisticRegression"):
			# Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join("test/resources/LogisticRegression_model.pkl"),"rb"))
            prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
            if prediction == 2:
                st.success('Sentiment prediction: 2 (Very Postive Sentiment)')
            if prediction == 1:
                st.success('Sentiment prediction: 1 (Postive Sentiment)')
            if prediction == 0:
                st.success('Sentiment prediction: 0 (Neutral Sentiment)')
            if prediction == -1:
                st.success('Sentiment prediction: -1 (Negative Sentiment)')

        if st.button("Classifier: KNeighborsClassifier"):
			# Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join("test/resources/KNeighborsClassifier_model.pkl"),"rb"))
            prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
            if prediction == 2:
                st.success('Sentiment prediction: 2 (Very Postive Sentiment)')
            if prediction == 1:
                st.success('Sentiment prediction: 1 (Postive Sentiment)')
            if prediction == 0:
                st.success('Sentiment prediction: 0 (Neutral Sentiment)')
            if prediction == -1:
                st.success('Sentiment prediction: -1 (Negative Sentiment)')


        if st.button("Classifier: PassiveAggressiveClassifier"):
			# Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join("test/resources/PassiveAggressiveClassifier_model.pkl"),"rb"))
            prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
            if prediction == 2:
                st.success('Sentiment prediction: 2 (Very Postive Sentiment)')
            if prediction == 1:
                st.success('Sentiment prediction: 1 (Postive Sentiment)')
            if prediction == 0:
                st.success('Sentiment prediction: 0 (Neutral Sentiment)')
            if prediction == -1:
                st.success('Sentiment prediction: -1 (Negative Sentiment)')

        if st.button("Classifier: GradientBoostingClassifier"):
			# Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join("test/resources/GradientBoostingClassifier_model.pkl"),"rb"))
            prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
            if prediction == 2:
                st.success('Sentiment prediction: 2 (Very Postive Sentiment)')
            if prediction == 1:
                st.success('Sentiment prediction: 1 (Postive Sentiment)')
            if prediction == 0:
                st.success('Sentiment prediction: 0 (Neutral Sentiment)')
            if prediction == -1:
                st.success('Sentiment prediction: -1 (Negative Sentiment)')

        st.subheader("**_Linear Support Vector Classifier_**")
        st.markdown("""
				A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane. In other words, given labeled training data (supervised learning),
				the algorithm outputs an optimal hyperplane which categorizes new examples.
				Since it is a linear svc the hyperplane will be predicting each class with a linear kernal.
				It is vary similar to SVC with kernal = 'linear' and the multiclass support is handled according to a one-vs-the-rest scheme.</p>
					""",unsafe_allow_html=True)
        st.info('Linear SVC for Multiclass')
		#image
        st.subheader("**_Logistic Regression Classifier_**")
        st.markdown("""
					It is a statistical method for analysing a data set in which there are one or more independent variables that determine an outcome.
					The outcome is measured with a dichotomous variable (in which there are only two possible outcomes).
					The goal of logistic regression is to find the best fitting model to describe the relationship between the dichotomous characteristic of
					interest (dependent variable = response or outcome variable) and a set of independent (predictor or explanatory) variables. This is better than other binary classification like nearest neighbor since it also
					explains quantitatively the factors that lead to classification.
					""",unsafe_allow_html=True)
        st.info('Linear Regression for Multiclass')
		#image

        st.subheader("**_Passive Agressive Classifier_**")
        st.markdown("""
				The goal is to find a hyperplane that seperates all the sentiment classes, on each round of analysing an
				observation and makes a prediction on the current hypothesis.
				It then compares prediction to true y and suffers a loss based on the difference.
				The goal is to make cumulative loss as small as possible.
				Finally, the hypothesis gets updated according to previous hypothesis and rhe current example.
					""",unsafe_allow_html=True)
        st.info('Passive Agressive Classifier for Binary Classes')
		#image

        st.subheader("**_K Nearest Neighbours Classifier_**")
        st.markdown("""
				K-nearest neighbors (KNN) algorithm uses 'feature similarity' to predict the values of new
				datapoints which further means that the new data point will be assigned a value based on how closely it
				matches the points in the training set. An object is classified by a majority vote of its neighbors, with the object
				being assigned to the class most common among its k nearest neighbors.
					""",unsafe_allow_html=True)
        st.info('K Nearest Neighbours Classifier for Multiclass')
		#image

        st.subheader("**_Gradient Boost Classifier_**")
        st.markdown("""
					<p>Gradient boosting is a type of machine learning boosting. It relies on the intuition that the best possible next model,
					when combined with previous models, minimizes the overall prediction error. The key idea is to set the target outcomes for
					this next model in order to minimize the error.</p>
					""",unsafe_allow_html=True)
        st.info('Gradient Boosting Classifier for Multiclass')
		#image





    # Our products and services pages
    if selection == "Our Products and Services":
        st.image("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/test/resources/products_services.png",
                  use_column_width= True)

        st.header("Our Products")
        st.markdown("""
                        - Natural Language Process
                        - Regression and Classifications Modeling
                        - Database Design
                        - Intelligent Dash-boarding
                        - Regression
                    """)
        st.header("Contact Us")
        st.markdown("If you wish to contact us please enter your message along with your contact details and we will get back to as soon as possible")
        st.text_area("Enter a message")
        #st.button("Submit")
        def func():
            st.write('Submitted Thank You')
            return
        if st.button("Submitted"):
            func()


    # About the Authors
    if selection == "About Us":
        st.image("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/test/resources/Aboutus%20(2).png",
                  use_column_width= True)
        st.title("OUR STORY")
        st.markdown("""
                    The Three Musketeers started in the Summer of 2019.
                    A group of individuals came together with the same vision:
                    **_Raise awareness for climate change within the way inviduals conduct business_**
                    A concept now brought to perfection by The Three Musketeers collective of creators.

                    Today, The Three Musketeers are helping businesses in determining consumer sentiment with regards to climate change. With these insights businesses will be able
                    to tailor the products and services to consumer satisfaction.

                    The Three Musketeers features a variety of models that businesses and consumers can make use of, using a broad spectrum of data preprocessing techniques and
                    hypertuning parameters.

                    Simply put: There's a model for every business, mindset and style.
                    """)

        st.title("MEET THE TEAM")

        st.header("Dominic Christopher Sadie")
        st.image("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/test/resources/IMG-20200628-WA0008%5B1%5D.jpg",
                 use_column_width= True)
        st.markdown("""
                    "I am a committed individual, who is constantly eager to learn. One of my strengths is my adaptability; I can develop new skills through self-study,
                    practice and learning from my team members."

                    **Background:**

                        - Studied electrical engineering at the University of Johannesburg.
                        - Volunteered with engineers without borders in Brazil: Helped implement solar panels
                                                                                in two schools.
                        - Worked as an intern at City Power: Learned how to manage asset.
                    """)

        st.header("Keamogetswe Makete")
        st.image("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/test/resources/IMG-20200626-WA0016%5B1%5D.jpg",
                  use_column_width= True)
        st.markdown(""" Risk taker: "Rather an opps than a what if"

                    **Background:**


                            - Bcom Honours in Economics
                    """)

        st.header("Nkululeko Mthembu")
        st.image("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/test/resources/IMG-20200628-WA0013%5B1%5D.jpg",
                  use_column_width= True)
        st.markdown("""
                    "Recommended by 9 out of 10 people who recommend things"

                    **Background:**

                            - A marketing enthusist with brand management background
                    """)


        st.header("Rolivhuwa Malise")
        st.image("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/test/resources/IMG-20200628-WA0012%5B1%5D.jpg",
                  use_column_width= True)
        st.markdown("""
                    "I am so cool even ice cubes are so jealous of me"

                    **Background:**

                            - National Diploma in IT
                            - Junior analyst background
                    """)

        st.header("Suvarna Chetty")
        st.image("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/test/resources/IMG_20190627_102247%5B1%5D.jpg",
                  use_column_width= True)
        st.markdown("""
                    “I am ambitious and driven. I thrive on challenge and constantly set goals for myself, so I have something to strive toward. I’m not comfortable with settling,
                    and I’m always looking for an opportunity to do better and achieve greatness."

                    **Background:**

                        - Studied Bachelor of Accounting at the University of Witswatersrand.
                        - Working for Savris Trading CC: Gained experience in many diverse roles
                                                         and positions
                    """)

        st.header("Ntokoza Nkanyane")
        st.image("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/master/test/resources/Ntokozo.PNG",
                  use_column_width= True)
        st.markdown(""" Risk taker: "Rather an opps than a what if"

                    **Background:**

                            - Studied accounting through UNISA
                            - Co-founded Bokamos Agro-processin: Produces animal feed
                    """)








# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
	main()
