### Libraries used in making of the web application
# Streamlit dependencies
import streamlit as st
import joblib, os

# Data dependencies
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


### Loading the data
# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/developing/resources/train.csv")

### The main function where we will build the actual app
def main():
    ## Creating sidebar with selection box
    # Creating load data
    st.sidebar.subheader(":heavy_check_mark: Data is loaded")
    st.sidebar.text_input("link to data", "https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/developing/resources/train.csv")

	# Creating multiple pages
    st.sidebar.title("Menu")
    options = ["Homepage", "Overview", "Our Mission", "Data Exploration", "Models", "About The Authors"]
    selection = st.sidebar.radio("Please select a page", options)

    ## Building our pages
    # Homepage page
    if selection == "Homepage":
        st.image("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/developing/test/resources/homepage.jpg")

        # Creating about us box on the side sidebar
        st.sidebar.title("About")
        st.sidebar.info("""
                    This web application is maintained by Three Musketeers. You can learn more about us on the **About The Authors** tab.
                    """)
        st.sidebar.image("https://i.pinimg.com/originals/5c/99/6a/5c996a625282d852811eac0ee9e81fbe.jpg",
                          use_column_width= True)


    # Overview page
    if selection == "Overview":

        # Added a main heading to the page
        st.image("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/developing/test/resources/overview2%20(2).png",
                  use_column_width= True)

        # Describing climate climate
        st.header("**_The terms “climate change” and “global warming” are used interchangeably but they mean separate things_**")
        st.image("https://snowbrains.com/wp-content/uploads/2019/08/climate_change_buzzwords.jpg?w=640", use_column_width= True)

        st.subheader(":bulb: What is climate change? :bulb:")
        st.markdown("""
                    A significant, long-term changes in global or regional climate patterns, in particular a change apparent from the mid to late 20th century
                    onwards and attributed largely to the increased levels of atmospheric carbon dioxide produced by the use of fossil fuels.
                    """)

        st.markdown("""
                    The global climate is the connected system of sun, earth and oceans, wind, rain and snow, forests, deserts and savannas, and everything that
                    people do. The climate of a place, for example, Johannesburg can be described as it's rainfall, changing temperatures during the year and so on.

                    But the global climate is more than the "average" of the climated of specific places.
                    """)

        st.image("https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_400,h_300/https://warmheartworldwide.org/wp-content/uploads/2018/07/Climate-change-graphic-1.gif",
                  caption = 'A description of the global climate includes how, for example, the rising temperature of the Pacific Ocean feeds typhoons which increases wind speeds, heavier rainfall and cause more damage, but also shifts global ocean currents that melt the ice in the Arctic and Antarctica which slowly makes sea level rise until coastal cities will decrease in land size. /n It is this systemic connectedness that makes global climate change so important and so complicated.',
                  use_column_width = True)

        st.subheader(":thermometer: What is global warming? :thermometer:")
        st.markdown("""
                    Global warming is the slow increase in the avergae temperature of the earth's atmosphere because of an increased amount of energy(heat) striking the
                    Earth from the sun is being trapped in the atmosphere and not radiated out into space.
                    <br>
                    The earth's atmosphere has always acted like a greenhouse to capture the sun's heat, ensuring that the earth has enjoyed temperatures that permitted the
                    emergence of life forms as we know them. Without our atmospheric greenhouse the Earth would be very cold.
                    <br>
                    Today, we have the opposite problem. The problem is not that too little sun warmth is reaching the earth, but that too much is being trapped in our atmosphere.
                    So much heat is being kept inside greenhouse earth that the temperature of the earth is going up faster than at any previous time in history.
                    """)
        st.image("https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_698/https://warmheartworldwide.org/wp-content/uploads/2018/07/greenhouse-effect-image.png",
                  use_column_width = True)

        st.subheader(":sun: How does global warming drive climate change? :sun:")
        st.markdown("""
                    All systems in the global climate system are connected, adding heat energy causes the global climate as a whole to change. Two thirds of the world is covered
                    with ocean which heats up. When the ocean heats up, more water evaporates into clouds. Where storms like hurricanes and typhoons are forming, the result is
                    more energy-intensive storms. A warmer atmosphere makes glaciers and mountain snow packs, the Polar ice cap, and the great ice shield jutting off of Antarctica
                    melt raising sea levels.
                    <br>
                    Changes in temperature change the great patterns of wind that bring the monsoons in Asia and rain and snow around the world, making drought and unpredictable
                    weather more common. This is why scientists have stopped focusing just on global warming and now focus on the larger topic of climate change.
                    """)
        st.image("https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_510,h_411/https://warmheartworldwide.org/wp-content/uploads/2018/07/Climate-change-connections.gif",
                  use_column_width = True)

        st.subheader("Climate change impact")
        st.markdown("""
                    **Rising Sea Levels**
                    Climate change impacts rising sea levels. Average sea level around the world rose about 8 inches (20 cm) in the past 100 years; climate scientists expect it to
                    rise more and more rapidly in the next 100 years as part of climate change impacts.

                    Sea rise is expected entirely to submerge a number of small, island countries, and to flood coastal spawning grounds for many staple marine resources, as well
                    as low-lying capital cities, commercial agriculture, transportation and power generation infrastructure and tourism investments. For an interactive map of how
                    different sea levels will affect different coastal areas worldwide

                    **Melting Ice**
                    Projections suggest climate change impacts within the next 100 years, if not sooner, the world’s glaciers will have disappeared, as will the Polar ice cap, and
                    the huge Antarctic ice shelf, Greenland may be green again, and snow will have become a rare phenomenon at what are now the world’s most popular ski resorts.

                    **Torrential Downpours and more powerful storms**
                    While the specific conditions that produce rainfall will not change, climate change impacts the amount of water in the atmosphere and will increase producing
                    violent downpours instead of steady showers when it does rain.
                    Hurricanes and typhoons will increase in power, and flooding will become more common.

                    Torrential downpours and devastating storms will increase large-scale damage to fields, homes, businesses, transportation and power systems and industry in
                    countries without the financial or human capital resources to respond.

                    **Heatwaves and droughts**
                    Despite downpours in some places, droughts and prolonged heatwaves will become common. Rising temperatures are hardly surprising, although they do not mean that
                    some parts of the world will not “enjoy” record cold temperatures and terrible winter storms.
                    (Heating disturbs the entire global weather system and can shift cold upper air currents as well as hot dry ones. Single snowballs and snowstorms do not make
                    climate change refutations.)
                    Increasingly, however, hot, dry places will get hotter and drier, and places that were once temperate and had regular rainfall will become much hotter and much
                    drier.

                    Heatwaves and droughts will increase pressure on already fragile power, healthcare, water and sewage systems, as well as reducing countries’ ability to feed
                    themselves or export agricultural products. Heat will also become an increasingly important killer, especially of the very young and the old.

                    **Changing eco-systems**
                    As the world warms, entire ecosystems will move.
                    Already rising temperatures at the equator have pushed such staple crops as rice north into once cooler areas, many fish species have migrated long distances to
                    stay in waters that are the proper temperature for them.
                    In once colder waters, this may increase fishermen’s catches; in warmer waters, it may eliminate fishing; in many places, such as on the East Coast of the US,
                    it will require fishermen to go further to reach fishing grounds.
                    Farmers in temperate zones are finding drier conditions difficult for crops such as corn and wheat, and once prime growing zones are now threatened.
                    Some areas may see complete ecological change.

                    changing ecosystems seem to result almost exclusively in the loss of important food species, for example of fish and staple crops, and the increase of malign
                    species such as disease vectors.

                    **Reduced food security**
                    One of the most striking impacts of rising temperatures is felt in global agriculture, although these impacts are felt very differently in the largely temperate
                    developed world and in the more tropical developing world.
                    Different crops grow best at quite specific temperatures and when those temperatures change, their productivity changes significantly.
                    At the same time, global population models suggest that developing world will add 3 billion people by 2050 and that developing world food producers must double
                    staple food crop production by then simply to maintain current levels of food consumption.

                    Food security, already shaky, is crumbling under rising temperatures and related climate changes. Major staple crops are declining in productivity, while unlike in
                    the developed countries, there are no new, more tropical staples to move in to take their places. Rising population combined with declining productivity,
                    increasing incidence of drought and storms is increasingly leaving developing countries vulnerable of food shortfalls.

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

        st.subheader("Visual impacts of climate change")
        st.image("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/developing/test/resources/Impactsofclimatechange.jpg",
                  use_column_width = True)

        st.subheader("Impact on the developing world")
        st.markdown("""
                    Climate change affects the entire globe; its impacts are more pronounced in the developing world than in the developed world.
                    In fact, ironically, although most of the human activity that produces climate change occurs in the developed world, many of climate changes’ effects will
                    actually be beneficial in the developed world. In the short- and middle-term, for example, climate change will likely increase fish and agricultural yields
                    where populations are small and shrinking and productivity is highest.
                    Climate change’s impacts in the developing world will be almost exclusively negative, often terribly so.

                    As K. Smith tartly observed in 2008:
                    “The rich will find their world to be more expensive, inconvenient, uncomfortable, disrupted and colourless; in general, more unpleasant and unpredictable,
                    perhaps greatly so. The poor will die.”
                    """)

        st.subheader("What can we do in the developing world to slow down climate change?")
        st.markdown("""
                    Countries in the developing world can make two major contributions to slowing climate change:
                    They can pursue smart development, avoiding the worst mistakes of the developed world; and They can reduce – even reverse – their one major contribution to
                    climate change: unsustainable agriculture practices.
                    What can the developing world do to avoid the mistakes of the developed world?
                    Look first at the primary sources of the GHGs that cause global warming: Power generation (25%); industry (21%); transportation (14%); and buildings (6%)

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



    # Our Mission page
    if selection == "Our Mission":
        st.image("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/developing/test/resources/ourmission.png",
                  use_column_width= True)


    # Data Exploration
    if selection == "Data Exploration":
        st.image("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/developing/test/resources/dataexploration.png",
                  use_column_width= True)



    # Models
    if selection == "Models":
        st.image("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/developing/test/resources/Predictivemodel.png",
                  use_column_width= True)


    # About the Authors
    if selection == "About The Authors":
        st.image("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/developing/test/resources/Aboutus.png",
                  use_column_width= True)







# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
	main()
