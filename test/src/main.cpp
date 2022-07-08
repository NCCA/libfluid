#include "myfluidsystem.h"
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <tclap/CmdLine.h>

/// Dump our data from the fluid problem to a file
void dumpToGeo(const std::string &filebase,
               const std::vector<float3> &points,
               const std::vector<float3> &colour,
               const std::vector<float3> &velocity,
               const uint cnt)
{
    char fname[200];
    std::sprintf(fname, "%s.%04d.geo", filebase.c_str(), cnt);

    // we will use a stringstream as it may be more efficient
    std::stringstream ss;
    std::ofstream file;
    file.open(fname);
    if (!file.is_open())
    {
        std::cout << "Failed to open file " << fname << '\n';
        exit(EXIT_FAILURE);
    }
    // write header see here http://www.sidefx.com/docs/houdini15.0/io/formats/geo
    ss << "PGEOMETRY V5\n";
    ss << "NPoints " << points.size() << " NPrims 1\n";
    ss << "NPointGroups 0 NPrimGroups 1\n";
    // this is hard coded but could be flexible we have 1 attrib which is Colour
    ss << "NPointAttrib 2  NVertexAttrib 0 NPrimAttrib 2 NAttrib 0\n";
    // now write out our point attrib this case Cd for diffuse colour
    ss << "PointAttrib \n";
    // Define the colour attribute - default the colour to white
    ss << "Cd 3 float 1 1 1\n";
    // Define the velocity attribute - default is 0
    ss << "v 3 float 0 0 0\n";
    // now we write out the particle data in the format
    // x y z 1 (attrib so in this case colour)
    std::vector<float3>::const_iterator pit, cit, vit;
    for (pit = points.begin(), cit = colour.begin(), vit = velocity.begin(); pit != points.end(); ++pit, ++cit, ++vit)
    {
        // Write out the point coordinates and a "1" (not sure what this is for)
        ss << (*pit).x << " " << (*pit).y << " " << (*pit).z << " 1 ";
        // Output the attributes (colour then velocity)
        ss << "(" << (*cit).x << " " << (*cit).y << " " << (*cit).z << " " << (*vit).x << " " << (*vit).y << " " << (*vit).z << ")\n";
    }

    // now write out the index values
    ss << "PrimitiveAttrib\n";
    ss << "generator 1 index 1 location1\n";
    ss << "dopobject 1 index 1 /obj/AutoDopNetwork:1\n";
    ss << "Part " << points.size() << " ";
    for (size_t i = 0; i < points.size(); ++i)
    {
        ss << i << " ";
    }
    ss << " [0	0]\n";
    ss << "box_object1 unordered\n";
    ss << "1 1\n";
    ss << "beginExtra\n";
    ss << "endExtra\n";
    // dump string stream to disk;
    file << ss.rdbuf();
    file.close();
}

int main(int argc, char *argv[])
{
    try
    {
        // Setup our command line parsing arguments and switches
        TCLAP::CmdLine cmd("fluidtest: a command line application to generate a SPH dam break simulation using the libfluid library.\n\
        Example usage: bin/fluidtest -f geo/sim",
                           ' ', "0.1");
        TCLAP::ValueArg<unsigned int> numparticlesArg("n", "num", "Number of particles (default 10000)", false, 10000, "unsigned int");
        TCLAP::ValueArg<unsigned int> gridresArg("g", "grid", "Resolution of grid for spacial partitioning on one axis (default 16)", false, 16, "unsigned int");
        TCLAP::ValueArg<float> adhesionArg("a", "adhesion", "Fluid adhesion (default 0.1)", false, 0.1f, "float");
        TCLAP::ValueArg<float> viscosityArg("v", "viscosity", "Fluid viscosity (default 0.2)", false, 0.2f, "float");
        TCLAP::ValueArg<float> stensionArg("s", "tension", "Surface tension (default 1.0)", false, 1.0f, "float");
        TCLAP::ValueArg<float> dampArg("d", "damp", "Velocity collision damping (default 0.5)", false, 0.5f, "float");
        TCLAP::ValueArg<float> timestepArg("t", "timestep", "Simulation time step (default 0.002)", false, 0.002f, "float");
        TCLAP::ValueArg<unsigned int> substepArg("p", "substeps", "Number of substeps in each time step (default 1)", false, 1, "unsigned int");
        TCLAP::ValueArg<unsigned int> numstepArg("q", "numsteps", "Number of overall simulation steps (default 500)", false, 500, "unsigned int");
        TCLAP::ValueArg<std::string> fileArg("f", "filebase", "Base path and filename for output geometry files", true, "", "string");

        cmd.add(numparticlesArg);
        cmd.add(gridresArg);
        cmd.add(adhesionArg);
        cmd.add(viscosityArg);
        cmd.add(stensionArg);
        cmd.add(dampArg);
        cmd.add(timestepArg);
        cmd.add(substepArg);
        cmd.add(numstepArg);
        cmd.add(fileArg);

        // Parse the argv array.
        cmd.parse(argc, argv);

        // Create a fluid system
        MyFluidSystem splash;
        splash.setupDoubleDamBreak(numparticlesArg.getValue(),
                                   gridresArg.getValue(),
                                   adhesionArg.getValue(),
                                   viscosityArg.getValue(),
                                   stensionArg.getValue(),
                                   dampArg.getValue());

        // Set up an output structure
        std::vector<float3> points(numparticlesArg.getValue());
        std::vector<float3> colour(numparticlesArg.getValue());
        std::vector<float3> velocity(numparticlesArg.getValue());

        float dt = timestepArg.getValue();
        uint substeps = substepArg.getValue();

        for (uint i = 1; i <= numstepArg.getValue(); ++i)
        {
            std::cout << "---Step " << i << ", Timestep " << dt * float(i) << "\n";
            splash.advance(dt, substeps);
            splash.exportToData(points, colour, velocity);
            dumpToGeo(fileArg.getValue(), points, colour, velocity, i);
        }
    }
    catch (TCLAP::ArgException &e)
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
