
from shapely.geometry import Polygon
def bbox_iou(bboxA, bboxB):
    # compute the intersection over union of two bboxes

    # Format of the bboxes is [tly, tlx, bry, brx, ...], where tl and br
    # indicate top-left and bottom-right corners of the bbox respectively.

    # determine the coordinates of the intersection rectangle
    xA = max(bboxA[1], bboxB[1])
    yA = max(bboxA[0], bboxB[0])
    xB = min(bboxA[3], bboxB[3])
    yB = min(bboxA[2], bboxB[2])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both bboxes
    bboxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    bboxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)
    
    iou = interArea / float(bboxAArea + bboxBArea - interArea)
    
    # return the intersection over union value
    return iou

def shape_iou(coordsA, coordsB):
    print("COORDS A ", coordsA)
    print("COORDS B ", coordsB)
    coordsA_1 = coordsA[0]
    coordsA_2 = coordsA[1]
    coordsA_3 = coordsA[2]
    coordsA_4 = coordsA[3]

    coordsB_1 = coordsB[0]
    coordsB_2 = coordsB[1]
    coordsB_3 = coordsB[2]
    coordsB_4 = coordsB[3]

    a = Polygon([(coordsA_1[0], coordsA_1[1]), (coordsA_2[0], coordsA_2[1]), (coordsA_3[0], coordsA_3[1]), (coordsA_4[0], coordsA_4[1])])
    b = Polygon([(coordsB_1[0], coordsB_1[1]), (coordsB_2[0], coordsB_2[1]), (coordsB_3[0], coordsB_3[1]), (coordsB_4[0], coordsB_4[1])])
    return  a.intersection(b).area / a.union(b).area