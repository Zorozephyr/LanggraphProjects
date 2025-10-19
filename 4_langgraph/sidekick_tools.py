from playwright.async_api import async_playwright
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from dotenv import load_dotenv
import os
import requests
from langchain.agents import Tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_openai import AzureChatOpenAI
import httpx
import json
from geopy.geocoders import Nominatim
import folium
from folium import plugins



load_dotenv(override=True)

# 🔥 SET PROXY BYPASS FIRST - AT MODULE LEVEL
os.environ["NO_PROXY"] = ",".join(filter(None, [
    os.getenv("NO_PROXY",""),
    ".autox.corp.amdocs.azr",
    "chat.autox.corp.amdocs.azr",
    "localhost","127.0.0.1"
]))
os.environ["no_proxy"] = os.environ["NO_PROXY"]

pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"
serper = GoogleSerperAPIWrapper()

async def playwright_tools():
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    return toolkit.get_tools(), browser, playwright


def push(text: str):
    """Send a push notification to the user"""
    requests.post(pushover_url, data = {"token": pushover_token, "user": pushover_user, "message": text})
    return "success"


def get_file_tools():
    toolkit = FileManagementToolkit(root_dir="sandbox")
    return toolkit.get_tools()


def create_trip_map(locations_json: str, map_title: str = "Trip Planning Map") -> str:
    """
    Create a color-coded Google Map for trip planning.
    
    Args:
        locations_json: JSON string with locations. Format:
            [
                {"name": "Hotel", "address": "123 Main St, New York", "color": "blue"},
                {"name": "Restaurant", "address": "456 Park Ave, New York", "color": "red"},
                {"name": "Museum", "address": "789 5th Ave, New York", "color": "green"}
            ]
        map_title: Title for the map
    
    Returns:
        Path to saved HTML map file
    """
    try:
        locations = json.loads(locations_json)
        
        # Initialize geocoder
        geolocator = Nominatim(user_agent="sidekick_trip_planner")
        
        # Calculate center of map (average of all locations)
        latitudes = []
        longitudes = []
        
        # Create folium map
        trip_map = folium.Map(
            location=[40.7128, -74.0060],  # Default to NYC
            zoom_start=12,
            tiles="OpenStreetMap"
        )
        
        # Color mapping
        color_map = {
            "blue": "#1f77b4",
            "red": "#d62728",
            "green": "#2ca02c",
            "purple": "#9467bd",
            "orange": "#ff7f0e",
            "brown": "#8c564b",
            "pink": "#e377c2",
            "gray": "#7f7f7f",
            "yellow": "#bcbd22"
        }
        
        # Add markers for each location
        for i, loc in enumerate(locations):
            try:
                location = geolocator.geocode(loc["address"])
                if location:
                    latitudes.append(location.latitude)
                    longitudes.append(location.longitude)
                    
                    color = loc.get("color", "blue").lower()
                    hex_color = color_map.get(color, "#1f77b4")
                    
                    # Create custom HTML for marker
                    popup_text = f"""
                    <b>{loc['name']}</b><br>
                    {loc['address']}<br>
                    <small>Stop #{i+1}</small>
                    """
                    
                    # Add marker with custom color
                    folium.Marker(
                        location=[location.latitude, location.longitude],
                        popup=folium.Popup(popup_text, max_width=250),
                        tooltip=loc['name'],
                        icon=folium.Icon(
                            color=loc.get("color", "blue").lower(),
                            icon='info-sign'
                        )
                    ).add_to(trip_map)
                    
            except Exception as e:
                print(f"Could not geocode {loc['address']}: {str(e)}")
        
        # Fit map to bounds of all markers
        if latitudes and longitudes:
            trip_map.fit_bounds(
                [[min(latitudes), min(longitudes)], [max(latitudes), max(longitudes)]]
            )
        
        # Add distance matrix (optional line connections)
        coords = [[lat, lon] for lat, lon in zip(latitudes, longitudes)]
        if len(coords) > 1:
            folium.PolyLine(
                coords,
                color='gray',
                weight=2,
                opacity=0.5,
                popup='Route'
            ).add_to(trip_map)
        
        # Save map
        map_path = "sandbox/trip_map.html"
        trip_map.save(map_path)
        
        return f"✅ Trip map created successfully! Saved to {map_path}\n📍 Locations marked: {len(locations)}\n🗺️ Open the HTML file in a browser to view the interactive map."
        
    except Exception as e:
        return f"❌ Error creating map: {str(e)}"


async def other_tools():
    push_tool = Tool(name="send_push_notification", func=push, description="Use this tool when you want to send a push notification")
    file_tools = get_file_tools()

    tool_search = Tool(
        name="search",
        func=serper.run,
        description="Use this tool when you want to get the results of an online web search"
    )

    wikipedia = WikipediaAPIWrapper()
    wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)

    python_repl = PythonREPLTool()
    
    # 🗺️ NEW: Google Maps Trip Planning Tool
    maps_tool = Tool(
        name="create_trip_map",
        func=create_trip_map,
        description="""Create a color-coded interactive map for trip planning. 
        Takes a JSON array of locations with addresses and colors.
        Example input: '[{"name": "Hotel", "address": "Times Square, New York", "color": "blue"}, {"name": "Museum", "address": "Central Park, New York", "color": "red"}]'
        Colors available: blue, red, green, purple, orange, brown, pink, gray, yellow"""
    )
    
    return file_tools + [push_tool, tool_search, python_repl, wiki_tool, maps_tool]


def llmConnection()->AzureChatOpenAI:
    # NOW create httpx clients - AFTER NO_PROXY is set ✅
    http_client = httpx.Client(
        verify=r"C:\amdcerts.pem",  # Use corporate certs
        timeout=30.0
    )

    async_http_client = httpx.AsyncClient(
        verify=r"C:\amdcerts.pem",
        timeout=30.0
    )

    AUTOX_API_KEY  = os.getenv("AUTOX_API_KEY")
    NTNET_USERNAME = (os.getenv("NTNET_USERNAME") or "").strip()

    return AzureChatOpenAI(
        azure_endpoint="https://chat.autox.corp.amdocs.azr/api/v1/proxy",
        api_key=AUTOX_API_KEY,
        azure_deployment="gpt-4o-128k",
        model="gpt-4o-128k",
        temperature=0.1,
        openai_api_version="2024-08-01-preview",
        default_headers={"username": NTNET_USERNAME, "application": "testing-proxyapi"},
        http_client=http_client,
        http_async_client=async_http_client
    )

