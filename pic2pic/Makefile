CFLAGS = -std=c++11 -O3
objects = main.o deal_2dpts.o get_mesh.o

face_transfer: $(objects)
	g++ $(CFLAGS) -o face_transfer $(objects) -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect

main.o: main.cpp deal_2dpts.hpp get_mesh.hpp
	g++ $(CFLAGS) -c main.cpp

deal_2dpts.o: deal_2dpts.cpp deal_2dpts.hpp get_mesh.hpp
	g++ $(CFLAGS) -c deal_2dpts.cpp

get_mesh.o: get_mesh.cpp get_mesh.hpp
	g++ $(CFLAGS) -c get_mesh.cpp

clean:
	-rm face_transfer $(objects)
