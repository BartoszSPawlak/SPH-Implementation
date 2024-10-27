#define _USE_MATH_DEFINES
#include <..\freeglut-2.8.1\include\GL\glut.h>
#include <stdlib.h>
#include <cmath>
#define M_PI 3.14159265358979323846
#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <mutex>
using namespace std;
#include <..\eigen-3.3.7\Eigen\Dense>
using namespace Eigen;

//Colors
GLfloat WHITE[3] = { 1,1,1 };
GLfloat RED[3] = { 1,0,0 };
GLfloat GREEN[3] = { 0,1,0 };

//A Ball
class Ball {

public:
    GLfloat* color; // bylo prywatne
    double maxHeight, x, y, z, radius; // bylo prywatne
    Vector3d v, f, co_ordinates, acceleration;
    double rho, p;
    int cell = 0;
    vector<size_t> neighbours;

    Ball(double r, GLfloat* c, double h, double x, double z, Vector3d v, Vector3d f, double rho, double p) :
        radius(r), color(c), maxHeight(h), y(h), x(x), z(z), v(0, 0, 0), f(0, 0, 0), rho(0), p(0.f), co_ordinates(x, h, z), acceleration(0, 0, 0) {};
    Ball() {}
};

//A chekerboard
class Checkerboard {
    int displayListId;

public:
    int width;
    int depth;
    Checkerboard(int width, int depth) : width(width), depth(depth) {}
    double centerx() { return width / 2; }
    double centery() { return depth / 2; }
    int returnx() { return width; }
    int returny() { return depth; }
    void create() {
        displayListId = glGenLists(1);
        glNewList(displayListId, GL_COMPILE);
        glBegin(GL_QUADS);
        glNormal3d(0, 1, 0);
        for (int x = 0; x < width - 1; x++) {
            for (int z = 0; z < depth - 1; z++) {
                glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE,
                    (x + z) % 2 == 0 ? RED : WHITE);
                glVertex3d(x, 0, z);
                glVertex3d(x + 1, 0, z);
                glVertex3d(x + 1, 0, z + 1);
                glVertex3d(x, 0, z + 1);
            }
        }
        glEnd();
        glEndList();
    }
    void draw() {
        glCallList(displayListId);
    }
};

int checkerboardSize = 8;
Checkerboard checkerboard(checkerboardSize, checkerboardSize);

static vector<Ball> balls;

Vector3d G(.0f, 12 * -9.8f, .0f);

Vector3d v(0, 0, 0), f(0, 0, 0);

Vector3d fpress(0, 0, 0);
Vector3d fvisc(0, 0, 0);

class SPH
{
public:
    // "Particle-Based Fluid Simulation for Applications"
    // solver parameters
    
    float REST_DENS = 1000.f; // rest density
    float GAS_CONST = 2000.f; // const for equation of state
    double H = 0.6f; // kernel radius
    float HSQ = H * H; // radius^2 for optimization
    float MASS = 65.f; // assume all particles have the same mass
    float VISC = 250.f; // viscosity constant
    float DT = 0.008f; // integration timestep 0.0008f

    // smoothing kernels defined in Müller and their gradients
    double POLY6 = 315.f / (65.f * M_PI * pow(H, 9.f));
    double SPIKY_GRAD = -45.f / (M_PI * pow(H, 6.f));
    double VISC_LAP = 45.f / (M_PI * pow(H, 6.f));

    // simulation parameters
    float EPS = H; // boundary epsilon
    float BOUND_DAMPING = -0.5f;

    // interaction
    const static int MAX_PARTICLES = 2500;
    const static int DAM_PARTICLES = 50;
    const static int BLOCK_PARTICLES = 250;

    // rendering projection parameters
    const static int WINDOW_WIDTH = 800;
    const static int WINDOW_HEIGHT = 600;
    double VIEW_WIDTH = 1.5 * 800.f;
    double VIEW_HEIGHT = 1.5 * 600.f;

    vector< size_t > sortedIndexOfParticles;
    int numberOfTimeStamp = 0;

    size_t normalizedWidth = static_cast<size_t>(checkerboardSize / (H * 2));
    size_t normalizedLength = static_cast<size_t>(checkerboardSize / (H * 2));
    size_t normalizedHeight = static_cast<size_t>(checkerboardSize / (H * 2));

    size_t CellsNumber = static_cast<size_t>(normalizedWidth *
        normalizedLength *
        normalizedHeight);

    vector< vector < size_t > > Cells;
    vector< vector < size_t > > nearbyCells;

    vector< vector < size_t > > resize(vector< vector < size_t > > Cells, size_t CellsNumber)
    {
        for (int i = 0; i < CellsNumber; i++)
        {
            Cells.push_back(vector < size_t >());
        }
        return Cells;
    }

    /**
         * @brief The main idea of numbering is to use height layers.
         * The x-axis is equal to width.
         * The y-axis is equal to length.
         * The z-axis is equal to height.
     **/

    vector< size_t > SortIndex(vector< size_t > sortedIndexOfParticles, int i=1)
    {
        int temp, j;

        for (int i = 1; i < sortedIndexOfParticles.size(); i++)
        {
            temp = sortedIndexOfParticles[i];

            for (j = i - 1; j >= 0 && sortedIndexOfParticles[j] > temp; j--)
                sortedIndexOfParticles[j + 1] = sortedIndexOfParticles[j];

            sortedIndexOfParticles[j + 1] = temp;
        }
            return sortedIndexOfParticles; 
    }

    void insertPointsIntoCells()
    {
        for (size_t i = 0; i < CellsNumber; i++)
            Cells[i].clear();

        size_t BSize = balls.size();

        for(auto& i : sortedIndexOfParticles)
        {
            // The Formula is created manually using height layers approach
            auto widthOffset = static_cast<size_t>(balls[i].co_ordinates(0) / ((double)H * 2));
            size_t lengthOffset = static_cast<size_t>(balls[i].co_ordinates(2) / ((double)H * 2)) *
                normalizedWidth;
            size_t heightOffset = static_cast<size_t>(balls[i].co_ordinates(1) / ((double)H * 2)) *
                normalizedLength * normalizedWidth;

            size_t boxIndex = widthOffset + lengthOffset + heightOffset;
            balls[i].cell = boxIndex;
            Cells[boxIndex].push_back(i);
        }
        if(numberOfTimeStamp % 100 == 0)
            sortedIndexOfParticles=SortIndex(sortedIndexOfParticles);
    }

    /**
     * @brief The main method of search.
     * 1. Clear all neighbours;
     * 2. Put every point in box;
     * 3. Look for neighbour points for every point in every box;
     * 4. Look for neighbour points for every point in neighbour boxes;
     */
    void search()
    {
        // 1
        for (size_t i = 0; i < balls.size(); i++)
            balls[i].neighbours.clear();
        // 2
        insertPointsIntoCells();
        // 3
        for (size_t boxIndex = 0; boxIndex < Cells.size(); boxIndex++)
            for (size_t pointIndex = 0; pointIndex < Cells[boxIndex].size(); pointIndex++)
                for (size_t nearbyPointIndex = 0; nearbyPointIndex < Cells[boxIndex].size(); nearbyPointIndex++)
                    if (pointIndex != nearbyPointIndex)
                    {
                        Vector3d difference = balls[Cells[boxIndex][pointIndex]].co_ordinates -
                            balls[Cells[boxIndex][nearbyPointIndex]].co_ordinates;
                        if (difference.squaredNorm() <= pow(H, 2))
                            balls[Cells[boxIndex][pointIndex]].neighbours.push_back(Cells[boxIndex][nearbyPointIndex]);
                    }
        // 4
        for (size_t boxIndex = 0; boxIndex < Cells.size(); boxIndex++)
            for (size_t pointIndex = 0; pointIndex < Cells[boxIndex].size(); pointIndex++)
                for (size_t nearbyBoxIndex = 0; nearbyBoxIndex < nearbyCells[boxIndex].size(); nearbyBoxIndex++)
                    for (size_t nearbyPointIndex = 0; nearbyPointIndex < Cells[nearbyCells[boxIndex][nearbyBoxIndex]].size(); nearbyPointIndex++)
                    {
                        Vector3d difference = balls[Cells[boxIndex][pointIndex]].co_ordinates -
                            balls[Cells[nearbyCells[boxIndex][nearbyBoxIndex]][nearbyPointIndex]].co_ordinates;
                        //if (difference.squaredNorm() - pow(H, 2) <= DBL_EPSILON)
                        if (difference.squaredNorm() - pow(H, 2) <= EPS)
                            balls[Cells[boxIndex][pointIndex]]
                            .neighbours
                            .push_back(Cells[nearbyCells[boxIndex][nearbyBoxIndex]][nearbyPointIndex]);
                    }
    }

    /**
     * @brief This method returns array of components (width, length and height) for box index.
     */

    Vector3d getComponentsOfBoxIndex(const size_t boxIndex)
    {
        size_t boxWidth = boxIndex % normalizedWidth; 
        size_t boxHeight = (boxIndex - boxWidth) / (normalizedWidth * normalizedLength); 
        size_t boxLength = (boxIndex - boxWidth -
            boxHeight * normalizedWidth * normalizedLength) / normalizedWidth; 

        Vector3d components(boxWidth, boxLength, boxHeight);

        return components;
    }

    /**
     * @brief This method returns type of box (one of six), using its components (width, length and height).
     */

    string getBoxType(const Vector3d& components)
    {
        if ((components[0] == 0 || components[0] == normalizedWidth - 1) &&
            (components[1] == 0 || components[1] == normalizedLength - 1) &&
            (components[2] == 0 || components[2] == normalizedHeight - 1))
        {
            return "outerCorne";
        }

        if ((components[0] != 0 && components[0] != normalizedWidth - 1) &&
            (components[1] == 0 || components[1] == normalizedLength - 1) &&
            (components[2] != 0 && components[2] != normalizedHeight - 1))
        {
            return "outerCenter";
        }

        if ((components[1] == 0 || components[1] == normalizedLength - 1) &&
            (
                ((components[0] != 0 && components[0] != normalizedWidth - 1) &&
                    (components[2] == 0 || components[2] == normalizedHeight - 1)) ||
                ((components[0] == 0 || components[0] == normalizedWidth - 1) &&
                    (components[2] != 0 && components[2] != normalizedHeight - 1))
                )
            )
        {
            return "outerLongitual";
        }

        if ((components[0] == 0 || components[0] == normalizedWidth - 1) &&
            (components[1] != 0 && components[1] != normalizedLength - 1) &&
            (components[2] == 0 || components[2] == normalizedHeight - 1))
        {
            return "innerCorner";
        }

        if ((components[0] != 0 && components[0] != normalizedWidth - 1) &&
            (components[1] != 0 && components[1] != normalizedLength - 1) &&
            (components[2] != 0 && components[2] != normalizedHeight - 1))
        {
            return "innerCenter";
        }

        if ((components[1] != 0 && components[1] != normalizedLength - 1) &&
            (
                ((components[0] != 0 && components[0] != normalizedWidth - 1) &&
                    (components[2] == 0 || components[2] == normalizedHeight - 1)) ||
                ((components[0] == 0 || components[0] == normalizedWidth - 1) &&
                    (components[2] != 0 && components[2] != normalizedHeight - 1))
                )
            )
        {
            return "innerLongitual";
        }

        return "innerLongitual";
    }

    void addForTopLeft(const Vector3d& /*components*/,
        const size_t boxIndex)
    {
        // addRight
        nearbyCells[boxIndex].push_back(boxIndex + 1);
        // addBotom
        nearbyCells[boxIndex].push_back(boxIndex - normalizedWidth * normalizedLength);
        // addBottomRight
        nearbyCells[boxIndex].push_back(boxIndex - normalizedWidth * normalizedLength + 1);
    }

    void addForTopRight(const Vector3d& /*components*/,
        const size_t boxIndex)
    {
        // addLeft
        nearbyCells[boxIndex].push_back(boxIndex - 1);
        // addBotom
        nearbyCells[boxIndex].push_back(boxIndex - normalizedWidth * normalizedLength);
        // addBottomLeft
        nearbyCells[boxIndex].push_back(boxIndex - normalizedWidth * normalizedLength - 1);
    }

    mutex mut2;
    void addForBottomLeft(const Vector3d& /*components*/,
        const size_t boxIndex)
    {
        mut2.lock();
        // addRight
        nearbyCells[boxIndex].push_back(boxIndex + 1);
        // addTop
        nearbyCells[boxIndex].push_back(boxIndex + normalizedWidth * normalizedLength);
        // addTopRight
        nearbyCells[boxIndex].push_back(boxIndex + normalizedWidth * normalizedLength + 1); mut2.unlock();
    }

    mutex mut3;
    void addForBottomRight(const Vector3d& /*components*/,
        const size_t boxIndex)
    {
        mut3.lock();
        // addLeft
        nearbyCells[boxIndex].push_back(boxIndex - 1);
        // addTop
        nearbyCells[boxIndex].push_back(boxIndex + normalizedWidth * normalizedLength);
        // addTopLeft
        nearbyCells[boxIndex].push_back(boxIndex + normalizedWidth * normalizedLength - 1); mut3.unlock();
    }
    mutex mut4;
    void addForCenter(const Vector3d& /*components*/,
        const size_t boxIndex)
    {
        mut4.lock();
        // addRight
        nearbyCells[boxIndex].push_back(boxIndex + 1);
        // addLeft
        nearbyCells[boxIndex].push_back(boxIndex - 1);
        // addTop
        nearbyCells[boxIndex].push_back(boxIndex + normalizedWidth * normalizedLength);
        // addBottom
        nearbyCells[boxIndex].push_back(boxIndex - normalizedWidth * normalizedLength);
        // addTopRight
        nearbyCells[boxIndex].push_back(boxIndex + normalizedWidth * normalizedLength + 1);
        // addTopLeft
        nearbyCells[boxIndex].push_back(boxIndex + normalizedWidth * normalizedLength - 1);
        // addBottomRight
        nearbyCells[boxIndex].push_back(boxIndex - normalizedWidth * normalizedLength + 1);
        // addBottomLeft
        nearbyCells[boxIndex].push_back(boxIndex - normalizedWidth * normalizedLength - 1); mut4.unlock();
    }
    mutex mut5;
    void addForLeft(const Vector3d& /*components*/,
        const size_t boxIndex)
    {
        mut5.lock();
        // addRight
        nearbyCells[boxIndex].push_back(boxIndex + 1);
        // addTop
        nearbyCells[boxIndex].push_back(boxIndex + normalizedWidth * normalizedLength);
        // addBottom
        nearbyCells[boxIndex].push_back(boxIndex - normalizedWidth * normalizedLength);
        // addTopRight
        nearbyCells[boxIndex].push_back(boxIndex + normalizedWidth * normalizedLength + 1);
        // addBottomRight
        nearbyCells[boxIndex].push_back(boxIndex - normalizedWidth * normalizedLength + 1); mut5.unlock();
    }

    void addForRight(const Vector3d& /*components*/,
        const size_t boxIndex)
    {
        // addLeft
        nearbyCells[boxIndex].push_back(boxIndex - 1);
        // addTop
        nearbyCells[boxIndex].push_back(boxIndex + normalizedWidth * normalizedLength);
        // addBottom
        nearbyCells[boxIndex].push_back(boxIndex - normalizedWidth * normalizedLength);
        // addTopLeft
        nearbyCells[boxIndex].push_back(boxIndex + normalizedWidth * normalizedLength - 1);
        // addBottomLeft
        nearbyCells[boxIndex].push_back(boxIndex - normalizedWidth * normalizedLength - 1);
    }

    void addForTop(const Vector3d& /*components*/,
        const size_t boxIndex)
    {
        // addRight
        nearbyCells[boxIndex].push_back(boxIndex + 1);
        // addLeft
        nearbyCells[boxIndex].push_back(boxIndex - 1);
        // addBottom
        nearbyCells[boxIndex].push_back(boxIndex - normalizedWidth * normalizedLength);
        // addBottomRight
        nearbyCells[boxIndex].push_back(boxIndex - normalizedWidth * normalizedLength + 1);
        // addBottomLeft
        nearbyCells[boxIndex].push_back(boxIndex - normalizedWidth * normalizedLength - 1);
    }
    mutex mut1;
    void addForBottom(const Vector3d& /*components*/,
        const size_t boxIndex)
    {
        mut1.lock();
        // addRight
        nearbyCells[boxIndex].push_back(boxIndex + 1);
        // addLeft
        nearbyCells[boxIndex].push_back(boxIndex - 1);
        // addTop
        nearbyCells[boxIndex].push_back(boxIndex + normalizedWidth * normalizedLength);
        // addTopRight
        nearbyCells[boxIndex].push_back(boxIndex + normalizedWidth * normalizedLength + 1);
        // addTopLeft
        nearbyCells[boxIndex].push_back(boxIndex + normalizedWidth * normalizedLength - 1); mut1.unlock();
    }

    void addForBack(const size_t boxIndex)
    {
        vector < size_t >  nearbyCellsTmp = nearbyCells[boxIndex];
        nearbyCellsTmp.push_back(boxIndex);

        for (size_t i = 0u; i < nearbyCellsTmp.size(); i++) {
            nearbyCellsTmp[i] += normalizedWidth;
            nearbyCells[boxIndex].push_back(nearbyCellsTmp[i]);
        }
    }

    void addForFront(const size_t boxIndex)
    {
        vector<size_t> nearbyCellsTmp = nearbyCells[boxIndex];
        nearbyCellsTmp.push_back(boxIndex);

        for (size_t i = 0u; i < nearbyCellsTmp.size(); i++) {
            nearbyCellsTmp[i] -= normalizedWidth;
            nearbyCells[boxIndex].push_back(nearbyCellsTmp[i]);
        }
    }

    mutex mut;
    void addForMiddle(const size_t boxIndex)
    {
        mut.lock();
        vector < size_t > nearbyCellsTmp1 = nearbyCells[boxIndex];
        nearbyCellsTmp1.push_back(boxIndex);

        vector < size_t > nearbyCellsTmp2 = nearbyCells[boxIndex];
        nearbyCellsTmp2.push_back(boxIndex);

        for (size_t i = 0u; i < nearbyCellsTmp1.size(); i++) {
            nearbyCellsTmp1[i] -= normalizedWidth;
            nearbyCells[boxIndex].push_back(nearbyCellsTmp1[i]);
            nearbyCellsTmp2[i] += normalizedWidth;
            nearbyCells[boxIndex].push_back(nearbyCellsTmp2[i]);
        }
        mut.unlock();
    }

    void addNearbyBoxesFor(const bool isLeft,
        const bool isRight,
        const bool isTop,
        const bool isBottom,
        const Vector3d& components,
        const size_t boxIndex)
    {
        if (isLeft && isBottom) {
            addForBottomLeft(components, boxIndex);
            return;
        }

        if (isRight && isBottom) {
            addForBottomRight(components, boxIndex);
            return;
        }

        if (isLeft && isTop) {
            addForTopLeft(components, boxIndex);
            return;
        }

        if (isRight && isTop) {
            addForTopRight(components, boxIndex);
            return;
        }

        if (isLeft) {
            addForLeft(components, boxIndex);
            return;
        }

        if (isRight) {
            addForRight(components, boxIndex);
            return;
        }

        if (isBottom) {
            addForBottom(components, boxIndex);
            return;
        }

        if (isTop) {
            addForTop(components, boxIndex);
            return;
        }
    }

    /**
     * @brief This method defines for box its neighbours, using box conmonents and type.
     */

    void defineNearbyBoxes(const string boxType,
        const Vector3d& components,
        const size_t boxIndex)
    {
        bool isLeft = components[0] == 0;
        bool isRight = components[0] == normalizedWidth - 1;
        bool isBack = components[1] == 0;
        bool isBottom = components[2] == 0;
        bool isTop = components[2] == normalizedHeight - 1;

        if (boxType == "outerCorner")
        {
            addNearbyBoxesFor(isLeft, isRight, isTop, isBottom, components, boxIndex);
            isBack ? addForBack(boxIndex) : addForFront(boxIndex);
        }
        else
            if (boxType == "outerCenter")//near and far
            {
                addForCenter(components, boxIndex);
                isBack ? addForBack(boxIndex) : addForFront(boxIndex);
            }
            else
                if (boxType == "outerLongitual")// vertical edges
                {
                    addNearbyBoxesFor(isLeft, isRight, isTop, isBottom, components, boxIndex);
                    isBack ? addForBack(boxIndex) : addForFront(boxIndex);
                }
                else
                    if (boxType == "innerCorner")// left and right wall
                    {
                        addNearbyBoxesFor(isLeft, isRight, isTop, isBottom, components, boxIndex);
                        addForMiddle(boxIndex);
                    }
                    else
                        if (boxType == "innerCenter")// of 26 neighbours
                        {
                            addForCenter(components, boxIndex);
                            addForMiddle(boxIndex);
                        }
                        else
                            if (boxType == "innerLongitual")// horizontal edges
                            {
                                addNearbyBoxesFor(isLeft, isRight, isTop, isBottom, components, boxIndex);
                                addForMiddle(boxIndex);
                            }
    }

    /**
     * @brief This method classifies boxes by types and fills nearby boxes for every box.
    **/

    void findNearbyCells()
    {
        for (size_t cellIndex = 0; cellIndex < Cells.size(); cellIndex++)
        {
            const Vector3d cellComponents = getComponentsOfBoxIndex(cellIndex);
            string cellType = getBoxType(cellComponents);
            defineNearbyBoxes(cellType, cellComponents, cellIndex);
        }
    }

    vector < size_t > ResizeSortedIndexOfParticles(vector < size_t > sortedIndexOfParticles)
    {
        for (int i = 0; i < DAM_PARTICLES; i++)
        {
            sortedIndexOfParticles.push_back(i);
        }
        return sortedIndexOfParticles;
    }

    void InitSPH(void)
    {
        Cells = resize(Cells, CellsNumber);
        nearbyCells = resize(nearbyCells, CellsNumber);
        findNearbyCells();

        sortedIndexOfParticles=ResizeSortedIndexOfParticles(sortedIndexOfParticles);

        int j;

        for (float y = EPS; y < checkerboard.width; y += H)
            for (float x = 1; x <= checkerboard.width / 2.7; x += H - 0.1)
                for (float z = 1; z <= checkerboard.depth / 2.7; z += H - 0.1)
                    if (balls.size() < DAM_PARTICLES)
                    {
                        j = 0;
                        Ball b = Ball(0.3, GREEN, y, x, z, v, f, 0.f, 0.f);

                        balls.push_back(b);
                    }
        search();
    }

    void Integrate(void)
    {
        for (auto& p : sortedIndexOfParticles)
        {
            // forward Euler integration

            Vector3d prevAcceleration = balls[p].acceleration;

            if (std::abs(balls[p].rho) > 0.)
                balls[p].acceleration = balls[p].f / balls[p].rho;

            Vector3d prevVelocity = balls[p].v;

            balls[p].v += (prevAcceleration + balls[p].acceleration) / 2.0 * DT;

            if (balls[p].v.squaredNorm() > 3.0)
                balls[p].v = prevVelocity;

            balls[p].co_ordinates += prevVelocity * DT + prevAcceleration / 2.0 * DT * DT;

            double EPS = balls[p].radius;

            // enforce boundary conditions
            if (balls[p].co_ordinates(0) - EPS < 0.0f)
            {
                balls[p].v(0) *= BOUND_DAMPING;
                balls[p].co_ordinates(0) = EPS;
            }

            int x = checkerboard.width;
            int y = checkerboard.depth;

            if (balls[p].co_ordinates(0) + EPS > checkerboard.width-1)
            {
                balls[p].v(0) *= BOUND_DAMPING;
                balls[p].co_ordinates(0) = x - EPS-1;
            }
            if (balls[p].co_ordinates(2) - EPS < 0.0f)
            {
                balls[p].v(2) *= BOUND_DAMPING;
                balls[p].co_ordinates(2) = EPS;
            }
        
            if (balls[p].co_ordinates(2) + EPS > checkerboard.depth-1)
            {
                balls[p].v(2) *= BOUND_DAMPING;
                balls[p].co_ordinates(2) = y - EPS-1;
            }
            if (balls[p].co_ordinates(1) - EPS < 0.0f)
            {
                balls[p].v(1) *= BOUND_DAMPING;
                balls[p].co_ordinates(1) = EPS;
            }
        }
        search();  
    }

    double Rho(Ball pi)
    {
        double rho = 0;
    
        for (auto& pj : sortedIndexOfParticles)
        {
            Vector3d rij = balls[pj].co_ordinates - pi.co_ordinates;
            double r2 = rij.squaredNorm();

            if (r2 < HSQ)
            {
                // this computation is symmetric
                rho += MASS * POLY6 * pow(HSQ - r2, 3.f);
            }
        }
        return rho;
    }

    void ComputeDensityPressure(void)
    {
        Ball ball;
    
        for (auto& pi : sortedIndexOfParticles)
        {
            balls[pi].rho = 0.f;
        
            balls[pi].rho += Rho(balls[pi]);

            balls[pi].p = GAS_CONST * (balls[pi].rho - REST_DENS);
        }
    }

    void F(Ball pi)
    {
        for (auto& pj : pi.neighbours)
            {
                if (&pi == &balls[pj])
                    continue;

                Vector3d rij = balls[pj].co_ordinates - pi.co_ordinates;
                double r = rij.norm();

                if (r < H)
                {
                    // compute pressure force contribution
                    fpress += -rij.normalized() * MASS * MASS * (pi.p + balls[pj].p) / (pi.rho * balls[pj].rho) * SPIKY_GRAD * pow(H - r, 2.f);
                    // compute viscosity force contribution
                    fvisc += VISC * MASS * (balls[pj].v - pi.v) / balls[pj].rho * VISC_LAP * (H - r);
                }
            }
    }

    void ComputeForces(void)
    {
        for (auto& pi : sortedIndexOfParticles)
        {
            fpress(0) = 0;
            fpress(1) = 0;
            fpress(2) = 0;

            fvisc(0) = 0;
            fvisc(1) = 0;
            fvisc(2) = 0;

            F(balls[pi]);

            Vector3d fgrav = G * balls[pi].rho;
            balls[pi].f = fpress + fvisc + fgrav;
        }
    }
};

SPH sph;
void Update()
{
    sph.ComputeDensityPressure();
    sph.ComputeForces();
    sph.Integrate();
    sph.numberOfTimeStamp++;
    glutPostRedisplay();
}

void init()
{
    glEnable(GL_DEPTH_TEST);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, WHITE);
    glLightfv(GL_LIGHT0, GL_SPECULAR, WHITE);
    glMaterialfv(GL_FRONT, GL_SPECULAR, WHITE);
    glMaterialf(GL_FRONT, GL_SHININESS, 30);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    checkerboard.create();
}

void update(void) {
    Update();

    for (auto& ball : balls) {
        glPushMatrix();
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, ball.color);
        glTranslated(ball.co_ordinates(0), ball.co_ordinates(1), ball.co_ordinates(2));
        glutSolidSphere(ball.radius, 30, 30);
        glPopMatrix();
    }
}

void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    gluLookAt(12.0, 3.0, 0.0, checkerboard.centerx(), 0.0, checkerboard.centery(), 0.0, 1.0, 0.0);
    checkerboard.draw();
    update();
    glFlush();
    glutSwapBuffers();
}

void reshape(int w, int h)
{
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(50.0, w / (GLfloat)h, 1.0, 150.0);
    glMatrixMode(GL_MODELVIEW);
}

void timer(int v)
{
    glutPostRedisplay();
    glutTimerFunc(1000 / 60, timer, v);
}

int main(int argc, char* argv[])
{   
    sph.InitSPH();
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800, 600);
    glutInitWindowPosition(10, 10);
    glutCreateWindow("SPH Implementation");
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutIdleFunc(Update);
    glutTimerFunc(1, timer, 0);
    
    init();
    glutMainLoop();
    return 0;
}