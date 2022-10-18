import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Tuple
import google_streetview.api
import google_streetview.helpers
import numpy as np
import pandas as pd
import tqdm
from dotenv import load_dotenv
parser = argparse.ArgumentParser("Data generator from google streetview images.")
parser.add_argument("-n", help="Number of images per country.", required=True, type=int)
parser.add_argument("--max", help="Max number of api requests per image.", default=3)
parser.add_argument("--datadir", help="Data directory.", default="data")
args = parser.parse_args()
@dataclass
class City:
    latitude: float
    longitude: float
    radius: float = 2000.0
class Country:

    def __init__(self, name: str):
        self.name = name
        self.cities: List[City] = []
        self.images: List[str] = []

    def add_image(self, image_path: str):
        self.images.append(image_path)

    def get_num_images(self):
        return len(self.images)

    def add_city(self, city: City):
        self.cities.append(city)

    def generate_coordinates(self) -> Tuple[float, float]:
        city = random.choice(self.cities)
        random_latitude, random_longitude = random_location(city.latitude, city.longitude, city.radius)
        return random_latitude, random_longitude

    def __repr__(self):
        return f"{self.name} ({len(self.cities)} cities, {self.get_num_images()} images)"

def random_location(lat, lon, max_radius):
    def random_point_in_disk(max_radius):
        r = max_radius * np.random.rand() ** 0.5
        theta = np.random.rand() * 2 * np.pi
        return r * np.cos(theta), r * np.sin(theta)
    EarthRadius = 6371  # km
    OneDegree = EarthRadius * 2 * np.pi / 360 * 1000
    dx, dy = random_point_in_disk(max_radius)
    random_lat = lat + dy / OneDegree
    random_lon = lon + dx / (OneDegree * np.cos(lat * np.pi / 180))
    return random_lat, random_lon

def parse_dataset(world_cities_dataset: str, data_folder: str) -> List[Country]:
    def parse_row(row):
        country_name = row["country"]
        if country_name in country_names:
            countries[country_name].add_city(City(row["lat"], row["lng"]))
    world_data = pd.read_csv(world_cities_dataset)
    world_data.apply(parse_row, axis=1)
    return list(countries.values())

def download(countries: List[Country], num_images: int, max_api_requests: int, data_folder: str):
    for country in tqdm.tqdm(countries):
        for _ in range(num_images):
            download_image(country, max_api_requests, data_folder)

def download_image(country: Country, max_api_requests: int, data_folder: str):
    def get_image(country: Country, location: str, heading: str, pitch: str):
        params = [
            {
                "location": location,
                "radius": 1000,
                "heading": heading,
                # "fov": 100,
                "pitch": pitch,
                "key": '<GOOGLE_API_KEY>',
            }
        ]
        results = google_streetview.api.results(params)
        status = results.metadata[0]["status"]
        if status == "OK":
            results.download_links(f"{data_folder}/{country.name}")
            image_number = country.get_num_images() + 1
            country.add_image(f"{image_number}.jpg")
            os.rename(
                os.path.join(data_folder, country.name, "gsv_0.jpg"),
                os.path.join(data_folder, country.name, f"{image_number}.jpg"),
            )
            os.remove(os.path.join(data_folder, country.name, "metadata.json"))
        else:
            print("error")
        return status

    api_requests = 0
    status = ""
    heading = ['0', '90', '180', '270']
    pitch = ['-90', '90']
    while status != "OK" and api_requests < max_api_requests:
        latitude, longitude = country.generate_coordinates()
        location = f"{latitude:.6f}, {longitude:.6f}"
        for i in range(4):
            status = get_image(country, location, heading[i], 0)
            if status != "OK":
                break
        if status != "OK":
            continue
        for i in range(2):
            status = get_image(country, location, 0, pitch[i])
            if status != "OK":
                break
        with open(os.path.join(data_folder, country.name, "target.csv"),"a") as f:
            f.write(location+'\n')
        api_requests += 1

def main():
    load_dotenv()
    countries = parse_dataset(world_cities_dataset="worldcities.csv", data_folder=args.datadir)
    download(
        countries=countries, num_images=args.n, max_api_requests=args.max, data_folder=args.datadir,
    )

country_names = ['United States', 'Swaziland', 'Korea, South', 'Macedonia',
                 'Iceland', 'Turkey', 'Canada', 'Tunisia',
                 'Norway', 'Israel', 'Senegal', 'New Zealand',
                 'Sweden', 'Jordan', 'Mexico', 'Ghana',
                 'Finland', 'United Arab Emirates', 'Guatemala', 'Nigeria',
                 'Denmark', 'Kyrgyzstan', 'Colombia', 'Uganda',
                 'Ireland', 'India', 'Brazil', 'Kenya',
                 'United Kingdom', 'Sri Lanka', 'Ecuador', 'Botswana',
                 'Netherlands', 'Bangladesh', 'Peru', 'South Africa',
                 'Luxembourg', 'Bhutan', 'Bolivia', 'France',
                 'Belgium', 'Mongolia', 'Chile', 'Lesotho',
                 'Argentina', 'Laos', 'Australia', 'Ukraine'
                 'Andorra', 'Japan', 'Uruguay', 'San Marino',
                 'Spain', 'Philippines', 'Portugal', 'Thailand',
                 'Germany', 'Cambodia', 'Switzerland', 'Malaysia',
                 'Austria', 'Singapore', 'Italy', 'Indonesia',
                 'Malta', 'Greece', 'Bulgaria', 'Russia'
                 'Albania', 'Montenegro', 'Serbia', 'Romania',
                 'Croatia', 'Slovenia', 'Hungary', 'Slovakia',
                 'Czechia', 'Poland', 'Lithuania', 'Latvia',
                 'Estonia']

countries = {}
for country_name in country_names:
    countries[country_name] = Country(country_name)

main()
