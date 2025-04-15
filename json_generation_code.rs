pub fn generate_nodes() -> String {
    let mut clipboard_string: String = "\"systems\": [\n".to_string();
    for systemi in 0..100 {
        let mut system = format!(
            "{{
            \"id\": \"system_{systemi}\",
            \"visiblename\": \"System_{systemi}\",
            \"description\": \"none\",
            \"nodes\": [
        "
        );
        for nodei in 0..10 {
            system.push_str(
                format!(
                    "{{
                \"id\": \"node_{systemi}-{nodei}\",
                \"template\": \"default\",
                \"visiblename\": \"Node_{systemi}-{nodei}\"
            }},
            "
                )
                .as_str(),
            );
        }
        system.push_str(
            "]
    },",
        );
        clipboard_string.push_str(system.as_str());
    }
    clipboard_string
}

pub fn generate_edges() -> String {
    let mut clipboard_string: String = "\"edges\": [".to_string();
    for lhs_i in 0..100 {
        for rhs_i in 0..100 {
            if rand::random::<f32>() > 0.9 && lhs_i != rhs_i {
                clipboard_string.push_str(
                    format!(
                        "[
                    \"node_{lhs_i}-0\",
                    \"node_{rhs_i}-0\",
                    \"intrasystem\"
                ],"
                    )
                    .as_str(),
                );
            }
        }
    }
    clipboard_string
}

pub fn generate_wars() -> String {
    let mut clipboard_string: String = "".to_string();
    for lhs_i in 0..10 {
        for rhs_i in 0..10 {
            if lhs_i != rhs_i {
                clipboard_string.push_str(
                    format!(
                        "[
                    \"faction_{lhs_i}\",
                    \"faction_{rhs_i}\"
                ],"
                    )
                    .as_str(),
                );
            }
        }
    }
    clipboard_string
}

pub fn generate_factoryclasses() -> String {
    let mut clipboard_string: String = "".to_string();
    for i in 0..10 {
        let input_index = 9 - i;
        clipboard_string.push_str(
            format!(
                "{{
                    \"id\": \"factory_{i}\",
                    \"visiblename\": \"Factory_{i}\",
                    \"description\": \"none\",
                    \"inputs\": [
                        {{
                            \"resourcetype\": \"resource_{input_index}\",
                            \"contents\": 0,
                            \"rate\": 1000,
                            \"target\": 10000,
                            \"capacity\": 10000
                        }}
                    ],
                    \"outputs\": [
                        {{
                            \"resourcetype\": \"resource_{i}\",
                            \"contents\": 0,
                            \"rate\": 1000,
                            \"target\": 10000,
                            \"capacity\": 10000
                        }}
                    ]
                }},
                "
            )
            .as_str(),
        );
    }
    clipboard_string
}

pub fn generate_hangars() -> String {
    let mut hangarclasses: Vec<Value> = Vec::new();
    for i in 0..10 {
        let i9 = i + 10;
        hangarclasses.push(json!({
                "id": format!("hangar_{i}"),
                "visiblename": format!("Hangar_{i}"),
                "description": "none",
                "capacity": 10000,
                "target": 10000,
                "allowed": [
                    format!("heavy_fighter_{i}"),
                    format!("heavy_fighter_{i9}"),
                    format!("light_fighter_{i}"),
                    format!("heavy_fighter_{i9}")
                ],
                "ideal": {
                    format!("heavy_fighter_{i}"): 10,
                    format!("heavy_fighter_{i9}"): 10,
                    format!("light_fighter_{i}"): 10,
                    format!("heavy_fighter_{i9}"): 10
                },
                "launch_volume": 100,
                "launch_interval": 10,
                "propagate": true,
            }
        ))
    }
    serde_json::to_string_pretty(&hangarclasses).unwrap()
}

pub fn generate_shipyardnames() -> String {
    let mut shipnames: String = "".to_string();
    for i in 0..10 {
        shipnames.push_str(
            format!("\"factory_{i}\": [1.0, 1, 1],
            ",).as_str()
        )
    }
    shipnames
}

pub fn generate_saliencenames() -> String {
    let mut names: String = "".to_string();
    for i in 0..20 {
        names.push_str(
            format!("\"heavy_fighter_{i}\": 1.0,
            ",).as_str()
        )
    }
    for i in 0..20 {
        names.push_str(
            format!("\"light_fighter_{i}\": 1.0,
            ",).as_str()
        )
    }
    names
}

pub fn generate_shipclasses() -> String {
    let mut shipclasses: Vec<Value> = Vec::new();
    for i in 0..10 {
        let shipyardcount = 4 - (i / 2);
        shipclasses.push(json!({
                "id": format!("carrier_{i}"),
                "visiblename": format!("Carrier_{i}"),
                "description": "none",
                "shipflavor": "default",
                "basehull": 1000,
                "basestrength": 1000,
                "aiclass": "default",
                "hangarvol": 1000,
                "hangars": [
                    format!("hangar_{i}")
                ],
                "engines": ["default"],
                "repairers": ["default"],
                "shipyardclasslist": [
                    format!("shipyard_{shipyardcount}")
                ]
            }
        ))
    }
    for i in 0..10 {
        shipclasses.push(json!({
                "id": format!("warship_{i}"),
                "visiblename": format!("Warship_{i}"),
                "description": "none",
                "shipflavor": "default",
                "basehull": 500,
                "basestrength": 500,
                "aiclass": "default",
                "hangarvol": 500,
                "engines": ["default"],
                "repairers": ["default"],
            }
        ))
    }
    for i in 0..10 {
        shipclasses.push(json!({
                "id": format!("factoryship_{i}"),
                "visiblename": format!("Factoryship_{i}"),
                "description": "none",
                "shipflavor": "default",
                "basehull": 1000,
                "basestrength": 1000,
                "aiclass": "default",
                "hangarvol": 1000,
                "engines": ["default"],
                "repairers": ["default"],
                "factoryclasslist": [
                    format!("factory_{i}"),
                ]
            }
        ))
    }
    for i in 0..10 {
        shipclasses.push(json!({
            "id": format!("freighter_{i}"),
            "visiblename": format!("Freighter_{i}"),
            "description": "none",
            "shipflavor": "default",
            "basehull": 500,
            "basestrength": 500,
            "aiclass": "default",
            "hangarvol": 500,
            "stockpiles": [
                {
                    "contents": {},
                    "target": 10000,
                    "capacity": 10000,
                }
            ],
            "engines": ["default"],
            "repairers": ["default"],
        }))
    }
    for i in 0..20 {
        shipclasses.push(json!({
            "id": format!("heavy_fighter_{i}"),
            "visiblename": format!("Heavy_Fighter_{i}"),
            "description": "none",
            "shipflavor": "default",
            "basehull": 20,
            "basestrength": 20,
            "aiclass": "default",
            "hangarvol": 20,
            "engines": ["default"],
            "repairers": ["default"],
        }))
    }
    for i in 0..20 {
        shipclasses.push(json!({
            "id": format!("light_fighter_{i}"),
            "visiblename": format!("Light_Fighter_{i}"),
            "description": "none",
            "shipflavor": "default",
            "basehull": 10,
            "basestrength": 10,
            "aiclass": "default",
            "hangarvol": 10,
            "repairers": ["default"],
        }))
    }
    serde_json::to_string_pretty(&shipclasses).unwrap()
}