// SLIC.cpp: implementation of the SLIC class.
//===========================================================================
// This code implements the zero parameter superpixel segmentation technique
// described in:
//
//
//
// "SLIC Superpixels Compared to State-of-the-art Superpixel Methods"
//
// Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua,
// and Sabine Susstrunk,
//
// IEEE TPAMI, Volume 34, Issue 11, Pages 2274-2282, November 2012.
//
// https://www.epfl.ch/labs/ivrl/research/slic-superpixels/
//===========================================================================
// Copyright (c) 2013 Radhakrishna Achanta.
//
// For commercial use please contact the author:
//
// Email: firstname.lastname@epfl.ch
//===========================================================================

#include <stdio.h>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <fstream>
#include "SLIC.h"
#include <chrono>
#include <cstring>


#include "omp.h"

typedef chrono::high_resolution_clock Clock;

// For superpixels
const int dx4[4] = {-1, 0, 1, 0};
const int dy4[4] = {0, -1, 0, 1};
//const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
//const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

// For supervoxels
const int dx10[10] = {-1, 0, 1, 0, -1, 1, 1, -1, 0, 0};
const int dy10[10] = {0, -1, 0, 1, -1, -1, 1, 1, 0, 0};
const int dz10[10] = {0, 0, 0, 0, 0, 0, 0, 0, -1, 1};
#ifdef Local
const int threadNumber = 4;
#else
const int threadNumber = 64;
#endif
//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

SLIC::SLIC() {
    m_lvec = NULL;
    m_avec = NULL;
    m_bvec = NULL;

    m_lvecvec = NULL;
    m_avecvec = NULL;
    m_bvecvec = NULL;
}

SLIC::~SLIC() {
    if (m_lvec) delete[] m_lvec;
    if (m_avec) delete[] m_avec;
    if (m_bvec) delete[] m_bvec;


    if (m_lvecvec) {
        for (int d = 0; d < m_depth; d++) delete[] m_lvecvec[d];
        delete[] m_lvecvec;
    }
    if (m_avecvec) {
        for (int d = 0; d < m_depth; d++) delete[] m_avecvec[d];
        delete[] m_avecvec;
    }
    if (m_bvecvec) {
        for (int d = 0; d < m_depth; d++) delete[] m_bvecvec[d];
        delete[] m_bvecvec;
    }
}

//==============================================================================
///	RGB2XYZ
///
/// sRGB (D65 illuninant assumption) to XYZ conversion
//==============================================================================
void SLIC::RGB2XYZ(
        const int &sR,
        const int &sG,
        const int &sB,
        double &X,
        double &Y,
        double &Z) {
    double R = sR / 255.0;
    double G = sG / 255.0;
    double B = sB / 255.0;

    double r, g, b;

    if (R <= 0.04045) r = R / 12.92;
    else r = pow((R + 0.055) / 1.055, 2.4);
    if (G <= 0.04045) g = G / 12.92;
    else g = pow((G + 0.055) / 1.055, 2.4);
    if (B <= 0.04045) b = B / 12.92;
    else b = pow((B + 0.055) / 1.055, 2.4);

    X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
}

//===========================================================================
///	RGB2LAB
//===========================================================================
void SLIC::RGB2LAB(const int &sR, const int &sG, const int &sB, double &lval, double &aval, double &bval) {
    //------------------------
    // sRGB to XYZ conversion
    //------------------------
    double X, Y, Z;
    RGB2XYZ(sR, sG, sB, X, Y, Z);

    //------------------------
    // XYZ to LAB conversion
    //------------------------
    double epsilon = 0.008856;    //actual CIE standard
    double kappa = 903.3;        //actual CIE standard

    double Xr = 0.950456;    //reference white
    double Yr = 1.0;        //reference white
    double Zr = 1.088754;    //reference white

    double xr = X / Xr;
    double yr = Y / Yr;
    double zr = Z / Zr;

    double fx, fy, fz;
    if (xr > epsilon) fx = pow(xr, 1.0 / 3.0);
    else fx = (kappa * xr + 16.0) / 116.0;
    if (yr > epsilon) fy = pow(yr, 1.0 / 3.0);
    else fy = (kappa * yr + 16.0) / 116.0;
    if (zr > epsilon) fz = pow(zr, 1.0 / 3.0);
    else fz = (kappa * zr + 16.0) / 116.0;

    lval = 116.0 * fy - 16.0;
    aval = 500.0 * (fx - fy);
    bval = 200.0 * (fy - fz);
}

//===========================================================================
///	DoRGBtoLABConversion
///
///	For whole image: overlaoded floating point version
//===========================================================================
void SLIC::DoRGBtoLABConversion(
        const unsigned int *&ubuff,
        double *&lvec,
        double *&avec,
        double *&bvec) {
    int sz = m_width * m_height;
    lvec = new double[sz];
    avec = new double[sz];
    bvec = new double[sz];
#pragma omp parallel for num_threads(threadNumber)
    for (int j = 0; j < sz; j++) {
        int r = (ubuff[j] >> 16) & 0xFF;
        int g = (ubuff[j] >> 8) & 0xFF;
        int b = (ubuff[j]) & 0xFF;

        RGB2LAB(r, g, b, lvec[j], avec[j], bvec[j]);
    }
}

//==============================================================================
///	DetectLabEdges
//==============================================================================
void SLIC::DetectLabEdges(
        const double *lvec,
        const double *avec,
        const double *bvec,
        const int &width,
        const int &height,
        vector<double> &edges) {
    int sz = width * height;

    edges.resize(sz, 0);
    for (int j = 1; j < height - 1; j++) {
        for (int k = 1; k < width - 1; k++) {
            int i = j * width + k;

            double dx = (lvec[i - 1] - lvec[i + 1]) * (lvec[i - 1] - lvec[i + 1]) +
                        (avec[i - 1] - avec[i + 1]) * (avec[i - 1] - avec[i + 1]) +
                        (bvec[i - 1] - bvec[i + 1]) * (bvec[i - 1] - bvec[i + 1]);

            double dy = (lvec[i - width] - lvec[i + width]) * (lvec[i - width] - lvec[i + width]) +
                        (avec[i - width] - avec[i + width]) * (avec[i - width] - avec[i + width]) +
                        (bvec[i - width] - bvec[i + width]) * (bvec[i - width] - bvec[i + width]);

            //edges[i] = (sqrt(dx) + sqrt(dy));
            edges[i] = (dx + dy);
        }
    }
}

//===========================================================================
///	PerturbSeeds
//===========================================================================
void SLIC::PerturbSeeds(
        vector<double> &kseedsl,
        vector<double> &kseedsa,
        vector<double> &kseedsb,
        vector<double> &kseedsx,
        vector<double> &kseedsy,
        const vector<double> &edges) {
    const int dx8[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
    const int dy8[8] = {0, -1, -1, -1, 0, 1, 1, 1};

    int numseeds = kseedsl.size();

    for (int n = 0; n < numseeds; n++) {
        int ox = kseedsx[n];//original x
        int oy = kseedsy[n];//original y
        int oind = oy * m_width + ox;

        int storeind = oind;
        for (int i = 0; i < 8; i++) {
            int nx = ox + dx8[i];//new x
            int ny = oy + dy8[i];//new y

            if (nx >= 0 && nx < m_width && ny >= 0 && ny < m_height) {
                int nind = ny * m_width + nx;
                if (edges[nind] < edges[storeind]) {
                    storeind = nind;
                }
            }
        }
        if (storeind != oind) {
            kseedsx[n] = storeind % m_width;
            kseedsy[n] = storeind / m_width;
            kseedsl[n] = m_lvec[storeind];
            kseedsa[n] = m_avec[storeind];
            kseedsb[n] = m_bvec[storeind];
        }
    }
}

//===========================================================================
///	GetLABXYSeeds_ForGivenK
///
/// The k seed values are taken as uniform spatial pixel samples.
//===========================================================================
void SLIC::GetLABXYSeeds_ForGivenK(
        vector<double> &kseedsl,
        vector<double> &kseedsa,
        vector<double> &kseedsb,
        vector<double> &kseedsx,
        vector<double> &kseedsy,
        const int &K,
        const bool &perturbseeds,
        const vector<double> &edgemag) {
    int sz = m_width * m_height;
    double step = sqrt(double(sz) / double(K));
    int T = step;
    int xoff = step / 2;
    int yoff = step / 2;

    int n(0);
    int r(0);
    for (int y = 0; y < m_height; y++) {
        int Y = y * step + yoff;
        if (Y > m_height - 1) break;

        for (int x = 0; x < m_width; x++) {
            //int X = x*step + xoff;//square grid
            int X = x * step + (xoff << (r & 0x1));//hex grid
            if (X > m_width - 1) break;

            int i = Y * m_width + X;

            //_ASSERT(n < K);

            //kseedsl[n] = m_lvec[i];
            //kseedsa[n] = m_avec[i];
            //kseedsb[n] = m_bvec[i];
            //kseedsx[n] = X;
            //kseedsy[n] = Y;
            kseedsl.push_back(m_lvec[i]);
            kseedsa.push_back(m_avec[i]);
            kseedsb.push_back(m_bvec[i]);
            kseedsx.push_back(X);
            kseedsy.push_back(Y);
            n++;
        }
        r++;
    }
    if (perturbseeds) {
        PerturbSeeds(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, edgemag);
    }

}

//===========================================================================
///	PerformSuperpixelSegmentation_VariableSandM
///
///	Magic SLIC - no parameters
///
///	Performs k mean segmentation. It is fast because it looks locally, not
/// over the entire image.
/// This function picks the maximum value of color distance as compact factor
/// M and maximum pixel distance as grid step size S from each cluster (13 April 2011).
/// So no need to input a constant value of M and S. There are two clear
/// advantages:
///
/// [1] The algorithm now better handles both textured and non-textured regions
/// [2] There is not need to set any parameters!!!
///
/// SLICO (or SLIC Zero) dynamically varies only the compactness factor S,
/// not the step size S.
//===========================================================================
void SLIC::PerformSuperpixelSegmentation_VariableSandM(
        double *kseedsl,
        double *kseedsa,
        double *kseedsb,
        double *kseedsx,
        double *kseedsy,
        int *klabels,
        const int &STEP,
        const int &NUMITR,
        const int numk) {
    int sz = m_width * m_height;
    //double cumerr(99999.9);
    int numitr(0);

    //----------------
    int offset = STEP;
    cout << "offset " << offset << endl;
    if (STEP < 10) offset = STEP * 1.5;
    //----------------

#ifdef Timer
    double minCost0 = 0;
    double minCost1 = 0;
    double minCost2 = 0;
    double minCost3 = 0;
    double minCost4 = 0;
    double minCost5 = 0;
    double minCost6 = 0;
    auto startTime = Clock::now();

#endif

    double *sigmal = new double[numk];
    double *sigmaa = new double[numk];
    double *sigmab = new double[numk];
    double *sigmax = new double[numk];
    double *sigmay = new double[numk];
    int *clustersize = new int[numk];

    double **sigmalT = new double *[threadNumber];
    for (int i = 0; i < threadNumber; i++) {
        sigmalT[i] = new double[numk];
    }
    double **sigmaaT = new double *[threadNumber];
    for (int i = 0; i < threadNumber; i++) {
        sigmaaT[i] = new double[numk];
    }
    double **sigmabT = new double *[threadNumber];
    for (int i = 0; i < threadNumber; i++) {
        sigmabT[i] = new double[numk];
    }
    double **sigmaxT = new double *[threadNumber];
    for (int i = 0; i < threadNumber; i++) {
        sigmaxT[i] = new double[numk];
    }
    double **sigmayT = new double *[threadNumber];
    for (int i = 0; i < threadNumber; i++) {
        sigmayT[i] = new double[numk];
    }
    int **clustersizeT = new int *[threadNumber];
    for (int i = 0; i < threadNumber; i++) {
        clustersizeT[i] = new int[numk];
    }

    double *distxy = new double[sz];
    double *distlab = new double[sz];
    double *distvec = new double[sz];
    double *maxlab = new double[numk];

    double **maxlabT = new double *[threadNumber];
    for (int i = 0; i < threadNumber; i++) {
        maxlabT[i] = new double[numk];
    }
#ifdef Timer
    auto endTime = Clock::now();
    auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    minCost0 += compTime.count() / 1000.0;
#endif
    //TODO remove is ok ?
//    double *maxxy = new double[numk];
    for (int i = 0; i < numk; i++) {
        maxlab[i] = 100;
    }
    double invxywt = 1.0 / (STEP * STEP);//NOTE: this is different from how usual SLIC/LKM works

    while (numitr < NUMITR) {
        //------
        //cumerr = 0;
        numitr++;
        //------
#ifdef Timer
        startTime = Clock::now();
#endif
#pragma omp parallel for num_threads(threadNumber)
        for (int i = 0; i < sz; i++) {
            distvec[i] = DBL_MAX;
        }
        for (int n = 0; n < numk; n++) {
            int y1 = max(0, (int) (kseedsy[n] - offset));
            int y2 = min(m_height, (int) (kseedsy[n] + offset));
            int x1 = max(0, (int) (kseedsx[n] - offset));
            int x2 = min(m_width, (int) (kseedsx[n] + offset));

#pragma omp parallel for num_threads(threadNumber)
            for (int y = y1; y < y2; y++) {
                for (int x = x1; x < x2; x++) {
                    int i = y * m_width + x;
                    //_ASSERT( y < m_height && x < m_width && y >= 0 && x >= 0 );

                    double l = m_lvec[i];
                    double a = m_avec[i];
                    double b = m_bvec[i];
                    //TODO 这里distlab可能被相邻的8个聚类中心更新，但是更新完之后的值是最后一个聚类中心计算出来的，这样maxlab的更新就取决于n的循环顺序
                    distlab[i] = (l - kseedsl[n]) * (l - kseedsl[n]) +
                                 (a - kseedsa[n]) * (a - kseedsa[n]) +
                                 (b - kseedsb[n]) * (b - kseedsb[n]);

                    distxy[i] = (x - kseedsx[n]) * (x - kseedsx[n]) +
                                (y - kseedsy[n]) * (y - kseedsy[n]);

                    //------------------------------------------------------------------------
                    double dist =
                            distlab[i] / maxlab[n] + distxy[i] * invxywt;//only varying m, prettier superpixels
                    //double dist = distlab[i]/maxlab[n] + distxy[i]/maxxy[n];//varying both m and S
                    //------------------------------------------------------------------------



                    if (dist < distvec[i]) {
                        distvec[i] = dist;
                        klabels[i] = n;
                    }
                }
            }
        }
#ifdef Timer

        auto endTime = Clock::now();
        auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
        minCost1 += compTime.count() / 1000.0;
        //-----------------------------------------------------------------
        // Assign the max color distance for a cluster
        //-----------------------------------------------------------------

        startTime = Clock::now();
#endif
        if (0 == numitr) {
#pragma omp parallel for num_threads(threadNumber)
            for (int i = 0; i < threadNumber; i++)
                for (int j = 0; j < numk; j++)
                    maxlabT[i][j] = 1;
#pragma omp parallel for num_threads(threadNumber)
            for (int i = 0; i < numk; i++) {
                maxlab[i] = 1;
            }
        }

#pragma omp parallel for num_threads(threadNumber)
        for (int i = 0; i < sz; i++) {
            int id = omp_get_thread_num();
            maxlabT[id][klabels[i]] = max(maxlabT[id][klabels[i]], distlab[i]);
        }
#pragma omp parallel for num_threads(threadNumber)
        for (int i = 0; i < numk; i++)
            for (int j = 0; j < threadNumber; j++)
                maxlab[i] = max(maxlab[i], maxlabT[j][i]);

#ifdef Timer

        endTime = Clock::now();
        compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
        minCost3 += compTime.count() / 1000.0;


        startTime = Clock::now();
#endif
        //-----------------------------------------------------------------
        // Recalculate the centroid and store in the seed values
        //-----------------------------------------------------------------
#pragma omp parallel for num_threads(threadNumber)
        for (int i = 0; i < threadNumber; i++)
            for (int j = 0; j < numk; j++) {
                sigmalT[i][j] = sigmaaT[i][j] = sigmabT[i][j] = sigmaxT[i][j] = sigmayT[i][j] = clustersizeT[i][j] = 0;
            }
#pragma omp parallel for num_threads(threadNumber)
        for (int i = 0; i < numk; i++) {
            sigmal[i] = sigmaa[i] = sigmab[i] = sigmax[i] = sigmay[i] = clustersize[i] = 0;
        }
#pragma omp parallel for num_threads(threadNumber)
        for (int i = 0; i < sz; i++) {
            int id = omp_get_thread_num();
            //TODO klabels[j] < 0 ?
            //_ASSERT(klabels[j] >= 0);
            sigmalT[id][klabels[i]] += m_lvec[i];
            sigmaaT[id][klabels[i]] += m_avec[i];
            sigmabT[id][klabels[i]] += m_bvec[i];
            sigmaxT[id][klabels[i]] += (i % m_width);
            sigmayT[id][klabels[i]] += (i / m_width);
            clustersizeT[id][klabels[i]]++;
        }

#pragma omp parallel for num_threads(threadNumber)
        for (int i = 0; i < numk; i++)
            for (int j = 0; j < threadNumber; j++) {
                sigmal[i] += sigmalT[j][i];
                sigmaa[i] += sigmaaT[j][i];
                sigmab[i] += sigmabT[j][i];
                sigmax[i] += sigmaxT[j][i];
                sigmay[i] += sigmayT[j][i];
                clustersize[i] += clustersizeT[j][i];
            }
#ifdef Timer

        endTime = Clock::now();
        compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
        minCost5 += compTime.count() / 1000.0;


        startTime = Clock::now();
#endif
#pragma omp parallel for num_threads(threadNumber)
        for (int k = 0; k < numk; k++) {
            double inv = 1.0 / clustersize[k];
            kseedsl[k] = sigmal[k] * inv;
            kseedsa[k] = sigmaa[k] * inv;
            kseedsb[k] = sigmab[k] * inv;
            kseedsx[k] = sigmax[k] * inv;
            kseedsy[k] = sigmay[k] * inv;
        }
#ifdef Timer

        endTime = Clock::now();
        compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
        minCost6 += compTime.count() / 1000.0;
#endif
    }
#ifdef Timer

    cout << "minCost0 : " << minCost0 << endl;
    cout << "minCost1 : " << minCost1 << endl;
    cout << "minCost2 : " << minCost2 << endl;
    cout << "minCost3 : " << minCost3 << endl;
    cout << "minCost4 : " << minCost4 << endl;
    cout << "minCost5 : " << minCost5 << endl;
    cout << "minCost6 : " << minCost6 << endl;
#endif
}

//===========================================================================
///	SaveSuperpixelLabels2PGM
///
///	Save labels to PGM in raster scan order.
//===========================================================================
void SLIC::SaveSuperpixelLabels2PPM(
        char *filename,
        int *labels,
        const int width,
        const int height) {
    FILE *fp;
    char header[20];

    fp = fopen(filename, "wb");

    // write the PPM header info, such as type, width, height and maximum
    fprintf(fp, "P6\n%d %d\n255\n", width, height);

    // write the RGB data
    unsigned char *rgb = new unsigned char[(width) * (height) * 3];
    int k = 0;
    unsigned char c = 0;
    for (int i = 0; i < (height); i++) {
        for (int j = 0; j < (width); j++) {
            c = (unsigned char) (labels[k]);
            rgb[i * (width) * 3 + j * 3 + 2] = labels[k] >> 16 & 0xff;  // r
            rgb[i * (width) * 3 + j * 3 + 1] = labels[k] >> 8 & 0xff;  // g
            rgb[i * (width) * 3 + j * 3 + 0] = labels[k] & 0xff;  // b

            // rgb[i*(width) + j + 0] = c;
            k++;
        }
    }
    fwrite(rgb, width * height * 3, 1, fp);

    delete[] rgb;

    fclose(fp);

}

//===========================================================================
///	EnforceLabelConnectivity
///
///		1. finding an adjacent label for each new component at the start
///		2. if a certain component is too small, assigning the previously found
///		    adjacent label to this component, and not incrementing the label.
//===========================================================================
void SLIC::EnforceLabelConnectivity(
        const int *labels,//input labels that need to be corrected to remove stray labels
        const int &width,
        const int &height,
        int *nlabels,//new labels
        int &numlabels,//the number of labels changes in the end if segments are removed
        const int &K) //the number of superpixels desired by the user
{
//	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
//	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

    const int dx4[4] = {-1, 0, 1, 0};
    const int dy4[4] = {0, -1, 0, 1};

    const int sz = width * height;
    const int SUPSZ = sz / K;
    //nlabels.resize(sz, -1);
#ifdef Timer
    double minCost0 = 0;
    double minCost1 = 0;
    auto startTime = Clock::now();
#endif
    for (int i = 0; i < sz; i++) nlabels[i] = -1;
#ifdef Timer
    auto endTime = Clock::now();
    auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    minCost0 += compTime.count() / 1000.0;
    startTime = Clock::now();
#endif
    int label(0);
    int *xvec = new int[sz];
    int *yvec = new int[sz];
    int oindex(0);
    int adjlabel(0);//adjacent label
    for (int j = 0; j < height; j++) {
        for (int k = 0; k < width; k++) {
            if (0 > nlabels[oindex]) {
                nlabels[oindex] = label;
                //--------------------
                // Start a new segment
                //--------------------
                xvec[0] = k;
                yvec[0] = j;
                //-------------------------------------------------------
                // Quickly find an adjacent label for use later if needed
                //-------------------------------------------------------
                {
                    for (int n = 0; n < 4; n++) {
                        int x = xvec[0] + dx4[n];
                        int y = yvec[0] + dy4[n];
                        if ((x >= 0 && x < width) && (y >= 0 && y < height)) {
                            int nindex = y * width + x;
                            if (nlabels[nindex] >= 0) adjlabel = nlabels[nindex];
                        }
                    }
                }

                int count(1);
                for (int c = 0; c < count; c++) {
                    for (int n = 0; n < 4; n++) {
                        int x = xvec[c] + dx4[n];
                        int y = yvec[c] + dy4[n];

                        if ((x >= 0 && x < width) && (y >= 0 && y < height)) {
                            int nindex = y * width + x;

                            if (0 > nlabels[nindex] && labels[oindex] == labels[nindex]) {
                                xvec[count] = x;
                                yvec[count] = y;
                                nlabels[nindex] = label;
                                count++;
                            }
                        }

                    }
                }
                //-------------------------------------------------------
                // If segment size is less then a limit, assign an
                // adjacent label found before, and decrement label count.
                //-------------------------------------------------------
                if (count <= SUPSZ >> 2) {
                    for (int c = 0; c < count; c++) {
                        int ind = yvec[c] * width + xvec[c];
                        nlabels[ind] = adjlabel;
                    }
                    label--;
                }
                label++;
            }
            oindex++;
        }
    }
    numlabels = label;

    if (xvec) delete[] xvec;
    if (yvec) delete[] yvec;
#ifdef Timer
    endTime = Clock::now();
    compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    minCost1 += compTime.count() / 1000.0;
    cout << "minCost0 : " << minCost0 << endl;
    cout << "minCost1 : " << minCost1 << endl;
#endif


}

//===========================================================================
///	PerformSLICO_ForGivenK
///
/// Zero parameter SLIC algorithm for a given number K of superpixels.
//===========================================================================
void SLIC::PerformSLICO_ForGivenK(
        const unsigned int *ubuff,
        const int width,
        const int height,
        int *klabels,
        int &numlabels,
        const int &K,//required number of superpixels
        const double &m)//weight given to spatial distance
{

    auto startTime = Clock::now();

    vector<double> kseedsl(0);
    vector<double> kseedsa(0);
    vector<double> kseedsb(0);
    vector<double> kseedsx(0);
    vector<double> kseedsy(0);
    //--------------------------------------------------
    m_width = width;
    m_height = height;
    int sz = m_width * m_height;
    //--------------------------------------------------
    //if(0 == klabels) klabels = new int[sz];
    for (int s = 0; s < sz; s++) klabels[s] = -1;
    //--------------------------------------------------
    if (1)//LAB
    {
        DoRGBtoLABConversion(ubuff, m_lvec, m_avec, m_bvec);
    } else//RGB
    {
        m_lvec = new double[sz];
        m_avec = new double[sz];
        m_bvec = new double[sz];
        for (int i = 0; i < sz; i++) {
            m_lvec[i] = ubuff[i] >> 16 & 0xff;
            m_avec[i] = ubuff[i] >> 8 & 0xff;
            m_bvec[i] = ubuff[i] & 0xff;
        }
    }
    //--------------------------------------------------
    auto endTime = Clock::now();
    auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "p1 cost : " << compTime.count() / 1000 << " ms" << endl;


    startTime = Clock::now();
    bool perturbseeds(true);
    vector<double> edgemag(0);
    //计算梯度
    if (perturbseeds) DetectLabEdges(m_lvec, m_avec, m_bvec, m_width, m_height, edgemag);
    endTime = Clock::now();
    compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "p1.5 cost : " << compTime.count() / 1000 << " ms" << endl;

    startTime = Clock::now();
    //找出kseeds，并且移到周围3*3网格中edgemag最小的
    GetLABXYSeeds_ForGivenK(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, K, perturbseeds, edgemag);
    int numk = kseedsl.size();
    double *kdseedsl = new double[numk];
    double *kdseedsa = new double[numk];
    double *kdseedsb = new double[numk];
    double *kdseedsx = new double[numk];
    double *kdseedsy = new double[numk];
    for (int i = 0; i < numk; i++) {
        kdseedsl[i] = kseedsl[i];
        kdseedsa[i] = kseedsa[i];
        kdseedsb[i] = kseedsb[i];
        kdseedsx[i] = kseedsx[i];
        kdseedsy[i] = kseedsy[i];
    }
    endTime = Clock::now();
    compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "p2 cost : " << compTime.count() / 1000 << " ms" << endl;


    startTime = Clock::now();
    int STEP = sqrt(double(sz) / double(K)) + 2.0;//adding a small value in the even the STEP size is too small.
    //迭代10次


    PerformSuperpixelSegmentation_VariableSandM(kdseedsl, kdseedsa, kdseedsb, kdseedsx, kdseedsy, klabels, STEP, 10,
                                                numk);
    numlabels = kseedsl.size();
    endTime = Clock::now();
    compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "p3 cost : " << compTime.count() / 1000 << " ms" << endl;


    startTime = Clock::now();
    int *nlabels = new int[sz];
    EnforceLabelConnectivity(klabels, m_width, m_height, nlabels, numlabels, K);
#pragma omp parallel for num_threads(threadNumber)
    for (int i = 0; i < sz; i++)
        klabels[i] = nlabels[i];
    if (nlabels) delete[] nlabels;
    endTime = Clock::now();
    compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "p4 cost : " << compTime.count() / 1000 << " ms" << endl;
}

//===========================================================================
/// Load PPM file
///
/// 
//===========================================================================
void LoadPPM(char *filename, unsigned int **data, int *width, int *height) {
    char header[1024];
    FILE *fp = NULL;
    int line = 0;

    fp = fopen(filename, "rb");

    // read the image type, such as: P6
    // skip the comment lines
    while (line < 2) {
        fgets(header, 1024, fp);
        if (header[0] != '#') {
            ++line;
        }
    }
    // read width and height
    sscanf(header, "%d %d\n", width, height);

    // read the maximum of pixels
    fgets(header, 20, fp);

    // get rgb data
    unsigned char *rgb = new unsigned char[(*width) * (*height) * 3];
    fread(rgb, (*width) * (*height) * 3, 1, fp);

    *data = new unsigned int[(*width) * (*height) * 4];
    int k = 0;
    for (int i = 0; i < (*height); i++) {
        for (int j = 0; j < (*width); j++) {
            unsigned char *p = rgb + i * (*width) * 3 + j * 3;
            // a ( skipped )
            (*data)[k] = p[2] << 16; // r
            (*data)[k] |= p[1] << 8;  // g
            (*data)[k] |= p[0];       // b
            k++;
        }
    }

    // ofc, later, you'll have to cleanup
    delete[] rgb;

    fclose(fp);
}

//===========================================================================
/// Load PPM file
///
/// 
//===========================================================================
int CheckLabelswithPPM(char *filename, int *labels, int width, int height) {
    char header[1024];
    FILE *fp = NULL;
    int line = 0, ground = 0;

    fp = fopen(filename, "rb");

    // read the image type, such as: P6
    // skip the comment lines
    while (line < 2) {
        fgets(header, 1024, fp);
        if (header[0] != '#') {
            ++line;
        }
    }
    // read width and height
    int w(0);
    int h(0);
    sscanf(header, "%d %d\n", &w, &h);
    if (w != width || h != height) return -1;

    // read the maximum of pixels
    fgets(header, 20, fp);

    // get rgb data
    unsigned char *rgb = new unsigned char[(w) * (h) * 3];
    fread(rgb, (w) * (h) * 3, 1, fp);

    int num = 0, k = 0;
    for (int i = 0; i < (h); i++) {
        for (int j = 0; j < (w); j++) {
            unsigned char *p = rgb + i * (w) * 3 + j * 3;
            // a ( skipped )
            ground = p[2] << 16; // r
            ground |= p[1] << 8;  // g
            ground |= p[0];       // b

            if (ground != labels[k])
                num++;

            k++;
        }
    }

    // ofc, later, you'll have to cleanup
    delete[] rgb;

    fclose(fp);

    return num;
}

//===========================================================================
///	The main function
///
//===========================================================================
int main(int argc, char **argv) {
    unsigned int *img = NULL;
    int width(0);
    int height(0);

    LoadPPM((char *) "input_image.ppm", &img, &width, &height);
    if (width == 0 || height == 0) return -1;

    cout << "wi : " << width << "  he : " << height << endl;
    int sz = width * height;
    int *labels = new int[sz];
    int numlabels(0);
    SLIC slic;
    int m_spcount;
    double m_compactness;
    m_spcount = 200;
    m_compactness = 10.0;
    auto startTime = Clock::now();
    slic.PerformSLICO_ForGivenK(img, width, height, labels, numlabels, m_spcount,
                                m_compactness);//for a given number K of superpixels
    auto endTime = Clock::now();
    auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "Computing time=" << compTime.count() / 1000 << " ms" << endl;

    int num = CheckLabelswithPPM((char *) "check.ppm", labels, width, height);
    if (num < 0) {
        cout << "The result for labels is different from output_labels.ppm." << endl;
    } else {
        cout << "There are " << num << " points' labels are different from original file." << endl;
    }

    slic.SaveSuperpixelLabels2PPM((char *) "output_labels.ppm", labels, width, height);
    if (labels) delete[] labels;

    if (img) delete[] img;

    return 0;
}
