# with open('/home/nsp/Desktop/TFM_Natalia/PIROPO/Test/annotations_28/annotations_omni1A_test2_28.txt') as f:
#     lines = f.readlines()
#     print(lines)
#     steps = len(lines)
#
#     # Iterate over each of the images
#     for i in range(0, steps):
#         # Load image
#         x = lines[i].split()
#         print(x)

fichero = open('/home/nsp/Desktop/TFM_Natalia/PIROPO/Test/annotations_24/annotations_omni3A_test2_24.txt', 'r')
f = open('/home/nsp/Desktop/TFM_Natalia/PIROPO/Test/annotations_24/annotations_omni3A_test2_24_b.txt', 'w')
for linea in fichero:
    x = " ".join(linea.split())
    f.write(x + '\n')
f.close()
fichero.close()