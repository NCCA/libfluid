#include "myfluidsystem.h"
#include <string> 
#include <fstream>
#include <iostream>
#include <sstream>


/// Dump our data from the fluid problem to a file
void dumpToGeo(const std::vector<float3> &points, 
               const std::vector<float3> &colour,
               const uint cnt) {
    char fname[100];
    std::sprintf(fname,"geo/sph.%04d.geo", cnt);

    // we will use a stringstream as it may be more efficient
    std::stringstream ss;
    std::ofstream file;
    file.open(fname);
    if (!file.is_open()) {
        std::cout << "failed to Open file "<<fname<<'\n';
        exit(EXIT_FAILURE);
    }
    // write header see here http://www.sidefx.com/docs/houdini15.0/io/formats/geo
    ss << "PGEOMETRY V5\n";
    ss << "NPoints " << points.size() << " NPrims 1\n";
    ss << "NPointGroups 0 NPrimGroups 1\n";
    // this is hard coded but could be flexible we have 1 attrib which is Colour
    ss << "NPointAttrib 1  NVertexAttrib 0 NPrimAttrib 2 NAttrib 0\n";
    // now write out our point attrib this case Cd for diffuse colour
    ss <<"PointAttrib \n";
    // default the colour to white
    ss <<"Cd 3 float 1 1 1\n";    
    // now we write out the particle data in the format
    // x y z 1 (attrib so in this case colour)
    std::vector<float3>::const_iterator pit,vit;
    for(pit = points.begin(), vit = colour.begin(); pit != points.end(); ++pit, ++vit) 
    {
        // Write out the point coordinates and a "1" (not sure what this is for)
        ss<< (*pit).x <<" "<< (*pit).y <<" "<< (*pit).z << " 1 ";
        // Output the colour attribute (lets leave it white)
        ss<<"("<< (*vit).x <<" "<< (*vit).y <<" "<< (*vit).z <<")\n";
    }
  
    // now write out the index values
    ss<<"PrimitiveAttrib\n";
    ss<<"generator 1 index 1 location1\n";
    ss<<"dopobject 1 index 1 /obj/AutoDopNetwork:1\n";
    ss<<"Part "<<points.size()<<" ";
    for(size_t i=0; i<points.size(); ++i)
    {
      ss<<i<<" ";
    }
    ss<<" [0	0]\n";
    ss<<"box_object1 unordered\n";
    ss<<"1 1\n";
    ss<<"beginExtra\n";
    ss<<"endExtra\n";
    // dump string stream to disk;
    file<<ss.rdbuf();
    file.close();   
}

int main(int argc, char *argv[]) 
{
    if (argc < 3) {
        std::cerr << "Usage: fluid <numPoints> <resolution>\n";
        exit(0);
    }
    MyFluidSystem splash;
    splash.setup(std::atoi(argv[1]), std::atoi(argv[2]));

    // Set up an output structure
    std::vector<float3> points(std::atoi(argv[1]));
    std::vector<float3> velocity(std::atoi(argv[1]));
    std::vector<float3> colour(std::atoi(argv[1]));

    float dt = 0.002f;
    uint substeps = 1;
    for (uint i=0; i<2000; ++i) {
        dumpToGeo(points, colour, i);
        std::cout << "Timestep="<<dt * float(i+1) << "\n";
        splash.advance(dt, substeps);
        splash.exportToData(points, colour, velocity);        
    }

    return EXIT_SUCCESS;
}
