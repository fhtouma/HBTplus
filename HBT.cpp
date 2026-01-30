using namespace std;
#include <iostream>
#include <string>
#include <cstdlib>
#include <omp.h>

#include "src/datatypes.h"
#include "src/config_parser.h"
#include "src/snapshot.h"
#include "src/halo.h"
#include "src/subhalo.h"
#include "src/mymath.h"

int main(int argc, char **argv)
{
#ifdef _OPENMP
 //omp_set_nested(0);
 omp_set_max_active_levels(1); //max_active_level 0: no para; 1: single layer;
#endif
  int snapshot_start, snapshot_end;
  ParseHBTParams(argc, argv, HBTConfig, snapshot_start, snapshot_end);
  mkdir(HBTConfig.SubhaloPath.c_str(), 0755);
  HBTConfig.DumpParameters();

  SubhaloSnapshot_t subsnap;

  subsnap.Load(snapshot_start-1, SubReaderDepth_t::SrcParticles);

  Timer_t timer;
  ofstream time_log(HBTConfig.SubhaloPath+"/timing.log", fstream::out|fstream::app);
  time_log<<fixed<<setprecision(1);//<<setw(8);
  for(int isnap=snapshot_start;isnap<=snapshot_end;isnap++)
  {
	timer.Tick();
	ParticleSnapshot_t partsnap;
	partsnap.Load(isnap);
	subsnap.SetSnapshotIndex(isnap);

	HaloSnapshot_t halosnap;
    // 修改开始：根据开关决定读取 Halo 还是创建全局 Halo
    if (HBTConfig.IsCosmological)
    {  
    	halosnap.Load(isnap);
    }
    else
    {
        // 理想实验模式：不读取 FoF，将整个 Snapshot 当作一个 Halo
        halosnap.SetSnapshotIndex(isnap);
        halosnap.Halos.resize(1); // 只有一个 Halo
        HBTInt np = partsnap.GetNumberOfParticles();
        halosnap.Halos[0].Particles.resize(np);
        halosnap.TotNumberOfParticles = np;
        
        // 并行填充所有粒子 ID
        #pragma omp parallel for
        for (HBTInt i = 0; i < np; i++)
        {
            halosnap.Halos[0].Particles[i] = partsnap.GetParticleId(i);
        }
        cout << "Ideal Simulation Mode: Created 1 fake halo with " << np << " particles." << endl;
    }
    // 修改结束    
      
	timer.Tick();
	#pragma omp parallel
	{
	halosnap.ParticleIdToIndex(partsnap);
	subsnap.ParticleIdToIndex(partsnap);
	//subsnap.star_formation(); //add star formation code here
	#pragma omp master
	timer.Tick();
	subsnap.AssignHosts(halosnap);
	}
	subsnap.PrepareCentrals(halosnap);

	timer.Tick();
	subsnap.RefineParticles();
	timer.Tick();

	subsnap.MergeSubhalos();
	timer.Tick();

	subsnap.UpdateTracks();
// 	subsnap.ParticleIndexToId();//moved inside Save().
	timer.Tick();

	subsnap.Save();

	timer.Tick();
	time_log<<isnap;
	for(int i=1;i<timer.Size();i++)
	  time_log<<"\t"<<timer.GetSeconds(i);
	time_log<<endl;
	timer.Reset();
  }

  return 0;
}
