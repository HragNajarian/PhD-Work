import cdsapi

client = cdsapi.Client()

dataset = "reanalysis-era5-single-levels"

request = {
        'product_type':['reanalysis'],
        'data_format':'grib',
        "download_format": "unarchived",
        'variable':[
            '10m_u_component_of_wind','10m_v_component_of_wind','2m_dewpoint_temperature',
            '2m_temperature','land_sea_mask','mean_sea_level_pressure',
            'sea_ice_cover','sea_surface_temperature','skin_temperature',
            'snow_depth','soil_temperature_level_1','soil_temperature_level_2',
            'soil_temperature_level_3','soil_temperature_level_4','surface_pressure',
            'volumetric_soil_water_layer_1','volumetric_soil_water_layer_2','volumetric_soil_water_layer_3',
            'volumetric_soil_water_layer_4'
        ],
        YY,
        MM,
        DD,
        # 'area':[Nort, West, Sout, East],
        'time':[
            "00:00", "01:00", "02:00",
            "03:00", "04:00", "05:00",
            "06:00", "07:00", "08:00",
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00",
            "15:00", "16:00", "17:00",
            "18:00", "19:00", "20:00",
            "21:00", "22:00", "23:00"
            ],
    }

client.retrieve(dataset, request, 'ERA5-DATE1-DATE2-sl.grib')