# Dengue Early Warning System

**Chuang Lab, Taipei Medical University**

Dengue is a mosquito-borne disease, and its outbreaks tend to follow the weather: rain leaves standing water where mosquitoes breed, and heat speeds up both the mosquitoes and the virus. This project uses that link to forecast where dengue risk is likely to rise across Southeast Asia, province by province, before the cases arrive.

**[See the latest forecast on the project website →](https://chuang-lab-dmptd-tmu.github.io/dengue-fever-EWS)**

## What it does

Every month the system:

1. **Collects satellite data** for each of 162 provinces in 10 Southeast Asian countries: rainfall, air and land temperature, evaporation, and vegetation.
2. **Feeds it to a model** trained on 2011–2018 dengue case reports. The model looks at the past 12 months of weather in each province and its neighbours.
3. **Publishes a risk map** for this month and next month. A risk of 1 means a typical level for that province at that time of year, 2 means double, and 0.5 means half.

The website also explains what drives each forecast: which inputs matter most, and which provinces the model pays the most attention to.

A second arm of the project, forecasting at a much finer scale within Taiwan, is in development.

## What's in this repository

| Folder | What it holds |
|---|---|
| [dengue-infection-module](dengue-infection-module/) | Cleans and standardises dengue case data from OpenDengue and the Indonesian Ministry of Health. |
| [zonal-statistics-module](zonal-statistics-module/) | Pulls satellite data per province from Google Earth Engine (a copy of [EZGEE](https://github.com/chuang-lab-DMPTD-TMU/Google-Earth-Engine-Zonal-Statistics-App)). |
| [climate-projection-module](climate-projection-module/) | Prepares future climate scenarios (Shared Socioeconomic Pathways) for longer-range projections. |
| [machine-learning-module](machine-learning-module/) | Trains, tests, and explains the models: a graph neural network (STGNN), plus Random Forest and XGBoost baselines. |
| [site](site/) | The project website, built with Quarto. |

## Getting started

Clone with the satellite-data module included:

```bash
git clone --recurse-submodules https://github.com/chuang-lab-DMPTD-TMU/dengue-fever-EWS.git
```

(In a clone that already exists, run `git submodule update --init`.)

Then pick the guide for what you want to do:

- **[Running the pipelines locally](guides/running-locally.md)**: build the Docker container, train or evaluate a model, and the tested hardware.
- **[Monthly forecast and cluster operations](guides/monthly-forecast-and-cluster.md)**: how the live map is produced, model bundles, the Kubernetes cluster, and explanation plots.

## License

[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
