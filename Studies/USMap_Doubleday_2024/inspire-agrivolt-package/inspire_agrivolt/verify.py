import xarray as xr

def verify_dataset_gids(irradiance_ds: xr.Dataset):
    """
    verify that a dataset contains valid albedo entries for each gid (lat, lon pair). Raise ValueError if it doesnt and print gids to be recomputed.

    verified datasets pass silently.
    """
    # it will be worth checking all variables, if any are null then we say we need to recompute at that gid.
    ref_var = "albedo" # comes from nsrdb -> pysam -> outputs

    mask = merged_with_gids[ref_var].mean(dim="time").isnull()
    skipped_gids = merged_with_gids.where(mask.compute(), drop=False).gid

    # visual representation of what was skipped
    #skipped_gids.plot()

    gids_arr = skipped_gids.values
    gids_arr = gids_arr[~np.isnan(gids_arr)].astype(int)

    if len(gids_arr) > 0:
        raise ValueError(
            """Dataset albedo incomplete at the following gids:""",
            {gids_arr}
            )