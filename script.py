from spatialdata import read_zarr
from napari_spatialdata import Interactive

import napari
import debugpy

# Enable debugpy and wait for debugger
debugpy.listen(("localhost", 5678))
print("Waiting for debugger to attach...")
debugpy.wait_for_client()  # Optional: only proceed once the debugger is attached

# Start Napari
viewer = napari.Viewer()
viewer.window.add_plugin_dock_widget("ilastik-napari")
# sdata = read_zarr("/Users/arnedf/VIB/DATA/test_data_ilastik/sdata_transcriptomics.zarr")
sdata = read_zarr("/Users/arnedf/VIB/DATA/test_data_ilastik/sdata_multi_channel.zarr")
Interactive(sdata)
napari.run()
