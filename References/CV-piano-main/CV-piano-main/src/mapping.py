import cv2
import numpy as np
from scipy.spatial import distance


"""section formula"""
def get_sections(line , points):

    sected_coords= []

    x1= line[0][0]
    y1= line[0][1]
    x2= line[1][0]
    y2= line[1][1]

    for m in range(1, points+1):
        n = points+1 - m
        x = int((x2*m + x1*n) / (m+n))
        y = int((y2*m + y1*n) / (m+n))

        sected_coords.append([x, y])
    return sected_coords


"""Drawing mesh inside those rectangles"""
def draw_mesh(verts, image):

    verts = [(tuple(coord)) for coord in verts]
    # Draw the polygon
    cv2.line(image, verts[0], verts[1], (0, 255, 0), 2)
    cv2.line(image, verts[2], verts[3], (0, 255, 0), 2)

    # Define the number of rows and columns for the mesh
    rows, cols = 1, 24

    # columns
    line_ad = get_sections([verts[0] , verts[1]] , cols-1)
    line_bc = get_sections([verts[2] , verts[3]] , cols-1)

    line_ad.insert(0 , verts[0])
    line_ad.append(verts[1])

    line_bc.insert(0 , verts[2])
    line_bc.append(verts[3])


    polygons = []
    for pnt in range(cols):
        polygon = [line_ad[pnt] , line_ad[pnt + 1] , line_bc[pnt + 1] , line_bc[pnt]]
        polygons.append(polygon)


    return cols , polygons


def draw_over_image(image , cols , polygons):
    colors = [(255 , 0 , 0 ) , (0 , 255 , 0) , (0 , 0 , 255) , (0 , 0 ,0)]
    for pnt in range(cols):
        polygon = np.array(polygons[pnt], dtype=np.int32)
        cv2.polylines(image, [polygon], True, colors[1], 2)

    return image


def analyse(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur and adaptive thresholding to obtain binary image
    blur = cv2.GaussianBlur(gray, (7, 7), 1)
    canny = cv2.Canny(blur , 100, 200)
    cv2.imshow("canny" , canny)
    # Find contours in the binary image
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and keep the largest ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    shapes = []
    
    # Loop over the contours3
    for contour in contours:
        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.024 * peri, True)
	
        # If the contour has 4 corners (a rectangle)
        if len(approx) == 4:
            shapes.append(approx)
	    
    # Display the result
    try:
        count = 1
        mesh_coords = []
        rect1 = [list(arr.flatten()) for arr in shapes[0]]
        rect2 = [list(arr.flatten()) for arr in shapes[1]]
        rect1.sort()
        rect2.sort()

        coords_n_dist = []

        for i in rect1:
            for j in rect2:
                dist = distance.euclidean(i , j)
                coords_n_dist.append([dist , [i , j]])
        coords_n_dist.sort()
        mesh_coords.extend(coords_n_dist[0][1])
        for i in coords_n_dist[1:]:
            # threshold of 10 over y values of points
            if coords_n_dist[0][1][0][1] + 10 < i[1][0][1]  and coords_n_dist[0][1][1][1] + 10< i[1][1][1] :
                max = i[1]
                break
        mesh_coords.extend(max)
	
        print("mesh_coords :", mesh_coords)

    except:
        print("could not mesh")
        return 0 , []
    else:
        return draw_mesh(mesh_coords, image)

