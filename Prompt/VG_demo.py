import visual_genome_python_driver.src.local as vg

dataDir = r'D:\Dataset\VG\unzip_data'

# ids = vg.GetAllImageData(dataDir)
# print(len(ids))  # 108077
# print(ids[0])  # id: 1, coco_id: -1, flickr_id: -1, width: 800, url: https://cs.stanford.edu/people/rak248/VG_100K_2/1.jpg
#
# region_output = vg.GetAllRegionDescriptions(dataDir)
# print(region_output[0][0])  # id: 1382, x: 421, y: 57, width: 82, height: 139, phrase: the clock is green in colour, image: 1

# QA_output = vg.GetAllQAs(dataDir)
# print(QA_output[0][0])  # id: 986768, image: 1, question: What color is the clock?, answer: Green.

# vg.AddAttrsToSceneGraphs(dataDir=dataDir)
# vg.SaveSceneGraphsById(dataDir=dataDir, imageDataDir=dataDir+'/by-id/')

scene_graph = vg.GetSceneGraph(1, images=dataDir, imageDataDir=dataDir+'/by-id/')
pass
