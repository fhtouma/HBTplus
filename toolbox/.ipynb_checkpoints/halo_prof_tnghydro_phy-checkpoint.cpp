/*to compute the density, phy velocity profile of each halo out to RMAX in logrithmic bins.
 * the bin edges are [0, r1, r2, ...rn) where (r1, ...rn) are generated as logspace(RMIN, RMAX, NBIN).
 * output the count in each bin for each halo.
 */

#include <cmath>
#include <iostream>
#include <string>
#include <stdlib.h>

#include "../src/datatypes.h"
#include "../src/config_parser.h"
#include "../src/snapshot.h"
#include "../src/halo.h"
#include "../src/subhalo.h"
#include "../src/mymath.h"
#include "../src/linkedlist.h"
// #include "../src/geometric_tree.h"
#include "../src/io/apostle_io.h"

#define RMIN 10.0 // phy kpc
#define RMAX 10000.0
#define NBIN 100 // 99+1

#define USE_LL //algorithm: whether to use linkedlist or geotree for spatial search.
#define NDIV 256

#define NPART_MIN 1000 // >=1000 particles

struct HaloSize_t
{
    int GroupIndex;
    int n[NBIN][TypeMax];
    HBTReal vr[NBIN][TypeMax];
    HBTReal mfr[NBIN][TypeMax];
    HaloSize_t(){};
    void Compute(HBTxyz &cen, HBTxyz &vel_cen, ParticleSnapshot_t &snap, Linkedlist_t &ll, HBTReal scalefactor, HBTReal hz);
};
void BuildHDFHaloSize(hid_t &H5T_dtypeInMem, hid_t &H5T_dtypeInDisk);

HBTReal DlnX=logf(RMAX/RMIN)/(NBIN-1); //spacing in ln-space for r
void save(vector <HaloSize_t> &HaloSize, int isnap, int ifile=0, int nfiles=0);

int main(int argc, char **argv)
{
    if(argc!=3)
    {
        cerr<<"Usage: "<<endl;
        cerr<<" "<<argv[0]<<" [config_file] [snapshot_number]"<<endl;
        cerr<<"    If snapshot_number<0, then it's counted from final snapshot in reverse order"<<endl;
        cerr<<"    (i.e., FinalSnapshot=-1,... FirstSnapshot=-N)"<<endl;
        return 1;
    }
    HBTConfig.ParseConfigFile(argv[1]);
    int isnap=atoi(argv[2]);
    if(isnap<0) isnap=HBTConfig.MaxSnapshotIndex+isnap+1;
    Timer_t timer;
    
    Apostle::ApostleHeader_t SnapHeader;
    SnapHeader.Fill(isnap);
    cout<<"NumberOfFiles "<<SnapHeader.NumberOfFiles
        <<"\nBoxSize "<<SnapHeader.BoxSize
        <<"\nScaleFactor "<<SnapHeader.ScaleFactor
        <<"\nOmegaM0 "<<SnapHeader.OmegaM0
        <<"\nOmegaLambda0 "<<SnapHeader.OmegaLambda0
        <<"\nMass "<<SnapHeader.Mass[0]<<", "<<SnapHeader.Mass[1]<<", "<<SnapHeader.Mass[2]<<", "<<SnapHeader.Mass[3]
        <<"\nNumPart "<<SnapHeader.NumPart
        <<"\n"<<SnapHeader.NumPart[0]<<", "<<SnapHeader.NumPart[1]<<", "<<SnapHeader.NumPart[2]<<", "<<SnapHeader.NumPart[3]
        <<"\nNumPartTotal "<<SnapHeader.NumPartTotal
        <<"\n"<<SnapHeader.NumPartTotal[0]<<", "<<SnapHeader.NumPartTotal[1]<<", "<<SnapHeader.NumPartTotal[2]<<", "<<SnapHeader.NumPartTotal[3]<<", "
        <<"\nNumPartAll "<<SnapHeader.NumPartAll
        <<endl;
    
    timer.Tick();
    HaloSnapshot_t halosnap;
    halosnap.LoadPosVel(isnap);
    timer.Tick();
    cout<<"HaloSnapshot PosVel load done, spend "<<timer.GetSeconds()<<"s"<<endl;
    cout<<"HaloSize = "<<halosnap.size()<<endl;
    
    timer.Tick();
    vector <HaloSize_t> HaloSize(halosnap.size());
    int nhalo=0;
    #pragma omp parallel for schedule(dynamic, 1000)
    for(HBTInt grpid=0;grpid<HaloSize.size();grpid++){
        if(halosnap.Halos[grpid].GroupLen<NPART_MIN){
            HaloSize[grpid].GroupIndex=-1;
            continue;
        }
        nhalo++;
        HaloSize[grpid].GroupIndex = grpid;
        for(int ibin=0;ibin<NBIN;ibin++){
            fill(HaloSize[grpid].n[ibin], HaloSize[grpid].n[ibin]+TypeMax, 0);
            fill(HaloSize[grpid].vr[ibin], HaloSize[grpid].vr[ibin]+TypeMax, 0.0);
            fill(HaloSize[grpid].mfr[ibin], HaloSize[grpid].mfr[ibin]+TypeMax, 0.0);
        }
    }
    timer.Tick();
    cout<<"HaloSize initialize done, spend "<<timer.GetSeconds()<<"s"<<endl;
    cout<<"HaloSize grouplen >="<<NPART_MIN<<" = "<<nhalo<<endl;
    
    nhalo=0;
    for(HBTInt grpid=0;grpid<HaloSize.size();grpid++){
        if(HaloSize[grpid].GroupIndex>=0)
            nhalo++;
    }
    cout<<"nhalo="<<nhalo<<endl;
    
    for(int grpid=0;grpid<10;grpid+=1){
        cout<<HaloSize[grpid].GroupIndex<<", "
            <<halosnap.Halos[HaloSize[grpid].GroupIndex].GroupLen<<", "
            <<halosnap.Halos[HaloSize[grpid].GroupIndex].ComovingAveragePosition[0]<<", "
            <<halosnap.Halos[HaloSize[grpid].GroupIndex].PhysicalAverageVelocity[0]<<endl;
    }
    cout<<endl;
    for(int grpid=0;grpid<1000000;grpid+=100000){
        cout<<HaloSize[grpid].GroupIndex<<", "
            <<halosnap.Halos[HaloSize[grpid].GroupIndex].GroupLen<<", "
            <<halosnap.Halos[HaloSize[grpid].GroupIndex].ComovingAveragePosition[0]<<", "
            <<halosnap.Halos[HaloSize[grpid].GroupIndex].PhysicalAverageVelocity[0]<<endl;
    }
    
    timer.Tick();
    ParticleSnapshot_t partsnap;
    partsnap.Load(isnap, false);
    timer.Tick();
    cout<<"ParticleSnapshot load done, spend "<<timer.GetSeconds()<<"s"<<endl;
    
    for(int partindex=0;partindex<10;partindex++){
        cout<<partsnap.GetParticleType(partindex)<<", "
            <<partsnap.GetMass(partindex)<<", "
            <<partsnap.GetPhysicalVelocity(partindex)[0]<<", "
            <<partsnap.GetComovingPosition(partindex)[0]<<endl;
    }
    cout<<endl;
    for(HBTInt partindex=0;partindex<30792391009;partindex+=3079239100){
        cout<<partsnap.GetParticleType(partindex)<<", "
            <<partsnap.GetMass(partindex)<<", "
            <<partsnap.GetPhysicalVelocity(partindex)[0]<<", "
            <<partsnap.GetComovingPosition(partindex)[0]<<endl;
    }
    
    auto &Cosmology=partsnap.Cosmology;
    cout<<"partsnap size = "<<partsnap.size()<<endl;
    cout<<"Hz="<<Cosmology.Hz<<endl;
    cout<<"ScaleFactor"<<Cosmology.ScaleFactor<<endl;
    
    timer.Tick();
    SnapshotPos_t PartPos(partsnap);
    Linkedlist_t searcher(NDIV, &PartPos, HBTConfig.BoxSize, HBTConfig.PeriodicBoundaryOn);
    timer.Tick();
    cout<<"linked list compiled, built in "<<timer.GetSeconds()<<"s"<<endl;
    
    cout<<"haloprof compute start"<<endl;
    timer.Tick();
    #pragma omp parallel for schedule(dynamic,1000)
    for(int grpid=0;grpid<HaloSize.size();grpid++)
    {
        if(HaloSize[grpid].GroupIndex<0) continue;
        HaloSize[grpid].Compute(halosnap.Halos[grpid].ComovingAveragePosition, halosnap.Halos[grpid].PhysicalAverageVelocity, partsnap, searcher, Cosmology.ScaleFactor, Cosmology.Hz);
        if(grpid%100000==0){
            cout<<grpid<<" computed"<<endl;
        }
    }
    timer.Tick();
    cout<<"done, spend "<<timer.GetSeconds()<<"s----------------"<<endl;

    auto it=HaloSize.begin(), it_save=HaloSize.begin();
    for(;it!=HaloSize.end();++it){
        if(it->GroupIndex>=0){
            if(it!=it_save) *it_save=move(*it);
            ++it_save;
        }
    }
    HaloSize.resize(it_save-HaloSize.begin());
    save(HaloSize, isnap);
    cout<<"HaloSize = "<<HaloSize.size()<<endl;
    
    return 0;
}

class LogBinCollector_t: public ParticleCollector_t
{
public:
    int (*N)[TypeMax];
    HBTReal (*VR)[TypeMax];
    HBTReal (*MFR)[TypeMax];
    HBTxyz CEN, VEL_CEN;
    ParticleSnapshot_t &SNAP;
    HBTReal HZ;
    HBTReal SCALEFACTOR;
    
    LogBinCollector_t(int (*n)[TypeMax], HBTReal (*vr)[TypeMax], HBTReal (*mfr)[TypeMax], HBTxyz cen, HBTxyz vel_cen, ParticleSnapshot_t &snap, HBTReal scalefactor, HBTReal hz): N(n), VR(vr), MFR(mfr), CEN(cen), VEL_CEN(vel_cen), SNAP(snap), SCALEFACTOR(scalefactor), HZ(hz)
    {}
    void Collect(HBTInt index, HBTReal d2)
    {
        const int &partType = SNAP.GetParticleType(index);
        const HBTReal &partMass = SNAP.GetMass(index);
        const HBTxyz &partPos = SNAP.GetComovingPosition(index);
        const HBTxyz &partVel = SNAP.GetPhysicalVelocity(index);

        HBTReal dx = partPos[0] - CEN[0];
        HBTReal dy = partPos[1] - CEN[1];
        HBTReal dz = partPos[2] - CEN[2];
        dx = NEAREST(dx);
        dy = NEAREST(dy);
        dz = NEAREST(dz);
        HBTReal r_onepart = sqrt(dx*dx + dy*dy + dz*dz)*SCALEFACTOR;//phy
        
        if(r_onepart < RMAX){
            int ibin=ceilf(logf(r_onepart/RMIN)/DlnX);
            if(ibin<0) ibin=0;
            
            HBTReal dvx = partVel[0] - VEL_CEN[0];
            HBTReal dvy = partVel[1] - VEL_CEN[1];
            HBTReal dvz = partVel[2] - VEL_CEN[2];
            HBTReal vr_onepart = (dvx*dx + dvy*dy + dvz*dz)*SCALEFACTOR/r_onepart + HZ*r_onepart;
            
            N[ibin][partType]+=1;
            VR[ibin][partType]+=vr_onepart;
            MFR[ibin][partType]+=vr_onepart*partMass;
        }
    }
};

void HaloSize_t::Compute(HBTxyz &cen, HBTxyz &vel_cen, ParticleSnapshot_t &snap, Linkedlist_t &ll, HBTReal scalefactor, HBTReal hz)
{
    LogBinCollector_t collector(n, vr, mfr, cen, vel_cen, snap, scalefactor, hz);
    ll.SearchSphere(1.001*RMAX/scalefactor, cen, collector);
}

void save(vector <HaloSize_t> &HaloSize, int isnap, int ifile, int nfiles)
{
    string filename="/home/jlzhou/Work/Boundary/data_final/TNG300-1";
    if(ifile==0&&nfiles==0)
        filename=filename+"/haloprof_phy_"+to_string(isnap)+".hdf5";
    else
        filename=filename+"/haloprof_phy_"+to_string(isnap)+"."+to_string(ifile)+".hdf5";
    hid_t file=H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t dim_atom[]={1}, dims[]={HaloSize.size()};
    hid_t H5T_HaloSizeInMem, H5T_HaloSizeInDisk;
    BuildHDFHaloSize(H5T_HaloSizeInMem, H5T_HaloSizeInDisk);
    writeHDFmatrix(file, &nfiles, "NumberOfFiles", 1, dim_atom, H5T_NATIVE_INT);
    writeHDFmatrix(file, HaloSize.data(), "HostHalos", 1, dims, H5T_HaloSizeInMem, H5T_HaloSizeInDisk);
    vector <float> rbin;
    logspace(RMIN, RMAX, NBIN, rbin);
    dims[0]=rbin.size();
    writeHDFmatrix(file, rbin.data(), "RadialBins", 1, dims, H5T_NATIVE_FLOAT);
    H5Tclose(H5T_HaloSizeInDisk);
    H5Tclose(H5T_HaloSizeInMem);
    H5Fclose(file);
}

void BuildHDFHaloSize(hid_t &H5T_dtypeInMem, hid_t &H5T_dtypeInDisk)
{
    H5T_dtypeInMem=H5Tcreate(H5T_COMPOUND, sizeof (HaloSize_t));
    hsize_t dims[]={NBIN,TypeMax};
    hid_t H5T_intArray=H5Tarray_create2(H5T_NATIVE_INT, 2, dims);
    hid_t H5T_HBTRealArray=H5Tarray_create2(H5T_HBTReal, 2, dims);

    #define InsertMember(x,t) H5Tinsert(H5T_dtypeInMem, #x, HOFFSET(HaloSize_t, x), t)//;cout<<#x<<": "<<HOFFSET(HaloSize_t, x)<<endl
    InsertMember(GroupIndex, H5T_NATIVE_INT);
    InsertMember(n, H5T_intArray);
    InsertMember(vr, H5T_HBTRealArray);
    InsertMember(mfr, H5T_HBTRealArray);
    #undef InsertMember

    H5T_dtypeInDisk=H5Tcopy(H5T_dtypeInMem);
    H5Tpack(H5T_dtypeInDisk); //clear fields not added to save disk space

    H5Tclose(H5T_intArray);
    H5Tclose(H5T_HBTRealArray);
}
