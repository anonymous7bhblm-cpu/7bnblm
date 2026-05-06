"""Dataset path and metadata helpers."""
import os

DATA_ROOT = "./data"


def get_temporal_info(dataset_name):
    """Get temporal info."""
    return {
        "annotation_dir": get_annotation_dir(dataset_name),
        "video_data_dir": get_video_data_dir(dataset_name),
        "meta_data_dir": get_meta_data_dir(dataset_name),
    }

def get_annotation_dir(dataset_name):
    """Get annotation dir."""
    if dataset_name == "american_football":
        return os.path.join(DATA_ROOT, "sports/simrank/")
    elif dataset_name == "mouse":
        return os.path.join(DATA_ROOT, "animal/simrank/")
    elif dataset_name == "enigma":
        return os.path.join(DATA_ROOT, "industry/simrank/")
    elif dataset_name == "animal_kingdom":
        return os.path.join(DATA_ROOT, "animal_kingdom/simrank/")
    elif dataset_name == "uca":
        return os.path.join(DATA_ROOT, "uca/simrank/")
    elif dataset_name == "multisports":
        return os.path.join(DATA_ROOT, "MultiSports/simrank/")
    elif dataset_name == "dota":
        return os.path.join(DATA_ROOT, "DoTA/simrank/")
    elif dataset_name == "cholectrack20":
        return os.path.join(DATA_ROOT, "CholecTrack20/simrank/")
    elif dataset_name == "meccano":
        return os.path.join(DATA_ROOT, "MECCANO/simrank/")
    elif dataset_name == "egosurgery":
        return os.path.join(DATA_ROOT, "egosurgery/simrank/")

    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")


def get_video_data_dir(dataset_name):
    """Get video data dir."""
    # grounding
    if dataset_name == "american_football":
        return os.path.join(DATA_ROOT, "american_football/videos")
    elif dataset_name == "mouse":
        return os.path.join(DATA_ROOT, "mouse/videos")
    elif dataset_name == "animal_kingdom":
        return os.path.join(DATA_ROOT, "animal_kingdom/videos")
    elif dataset_name == "uca":
        return os.path.join(DATA_ROOT, "uca/videos")
    elif dataset_name == "multisports":
        return os.path.join(DATA_ROOT, "MultiSports/videos")
    elif dataset_name == "dota":
        return os.path.join(DATA_ROOT, "DoTA/videos")
    elif dataset_name == "cholectrack20":
        return os.path.join(DATA_ROOT, "CholecTrack20/videos")
    elif dataset_name == "meccano":
        return os.path.join(DATA_ROOT, "MECCANO/videos")
    elif dataset_name == "egosurgery":
        return os.path.join(DATA_ROOT, "egosurgery/videos")
    elif dataset_name == "enigma":
        return os.path.join(DATA_ROOT, "ENIGMA/videos")
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

def get_meta_data_dir(dataset_name):
    """Used for open QA datasets."""
    # grounding
    if dataset_name == "american_football":
        return os.path.join(DATA_ROOT, "american_football/meta-data")
    elif dataset_name == "mouse":
        return os.path.join(DATA_ROOT, "mouse/meta-data")
    elif dataset_name == "enigma":
        return os.path.join(DATA_ROOT, "ENIGMA/meta-data")
    elif dataset_name == "animal_kingdom":
        return os.path.join(DATA_ROOT, "animal_kingdom/meta-data")
    elif dataset_name == "uca":
        return os.path.join(DATA_ROOT, "uca/meta-data")
    elif dataset_name == "multisports":
        return os.path.join(DATA_ROOT, "MultiSports/meta-data")
    elif dataset_name == "dota":
        return os.path.join(DATA_ROOT, "DoTA/meta-data")
    elif dataset_name == "cholectrack20":
        return os.path.join(DATA_ROOT, "CholecTrack20/meta-data")
    elif dataset_name == "meccano":
        return os.path.join(DATA_ROOT, "MECCANO/meta-data")
    elif dataset_name == "egosurgery":
        return os.path.join(DATA_ROOT, "egosurgery/meta-data")
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")


def get_domain_from_dataset_name(dataset_name: str) -> str:
    """Return the domain for a dataset name."""
    if dataset_name in ["animal_kingdom", "mouse"]:
        domain = "animal"
    elif dataset_name in ["uca", "dota"]:
        domain = "safety"
    elif dataset_name in ["multisports", "american_football"]:
        domain = "sports"
    elif dataset_name in ["egosurgery", "cholectrack20"]:
        domain = "surgery"
    elif dataset_name in ["meccano", "enigma"]:
        domain = "industry"
    else:
        raise ValueError(f"invalid dataset_name: {dataset_name}")

    return domain
