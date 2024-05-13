#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <array>

float dist(float x1, float y1, float x2, float y2) { return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)); }
sf::Vector2f addVectors2f(sf::Vector2f v1, sf::Vector2f v2) { return sf::Vector2f(v1.x + v2.x, v1.y + v2.y); }
sf::Vector2f subVectors2f(sf::Vector2f v1, sf::Vector2f v2) { return sf::Vector2f(v1.x - v2.x, v1.y - v2.y); }
sf::Vector2f multVectors2f(sf::Vector2f v1, sf::Vector2f v2) { return sf::Vector2f(v1.x * v2.x, v1.y * v2.y); }
sf::Vector2f normalizeVector2f(sf::Vector2f v1)
{
    float d = dist(0, 0, v1.x, v1.y);
    return sf::Vector2f(v1.x / d, v1.y / d);
}
float dotProductVectors2f(sf::Vector2f v1, sf::Vector2f v2) { return v1.x * v2.x + v1.y * v2.y; };
float angleVect(sf::Vector2f v)
{
    if (v.x == 0)
    {
        return 3.141592 / 2 * abs(v.y) / v.y;
    };
    if (v.y == 0)
    {
        return 3.141592 + 3.141592 * abs(v.x) / v.x;
    };
    return atan2(v.y, v.x);
}
sf::Vector2f vectAngle(float a) { return sf::Vector2f(cos(a), sin(a)); }

void write(sf::RenderWindow& window, std::string t1, sf::Vector3f v, sf::Color col)
{
    // load font
    sf::Font font;
    font.loadFromFile("../fonts/Anton-Regular.ttf");

    // create the text element
    sf::Text text(t1, font);
    text.setStyle(sf::Text::Bold);
    text.setFillColor(col);

    text.setCharacterSize(v.z);

    text.setPosition(sf::Vector2f(v.x, v.y));

    window.draw(text);
}

class Layer
{
public:
    int size;
    int preSize;
    std::vector<float> values;
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;

    void init(int s, int ps);
    void randomize();
    // void copyLayerAtPrcnt(Layer L, int prcnt);
};

void Layer::init(int s, int ps)
{
    size = s;
    preSize = ps;
    values.clear();
    biases.clear();
    std::vector<float> v;
    for (int a = 0; a < ps; a++)
    {
        v.push_back(0);
    }

    for (int i = 0; i < s; i++)
    {
        values.push_back(0);
        biases.push_back(0);
        weights.push_back(v);
    }
}

void Layer::randomize()
{
    for (int i = 0; i < size; i++)
    {
        // values[i] = (rand()%1000-500)/100;
        biases[i] = (rand() % 1000 - 500) / 100;
        for (int a = 0; a < preSize; a++)
        {
            weights[i][a] = (rand() % 1000 - 500) / 200;
        }
    }
}

struct NeuralNet
{
    std::vector<Layer> layers;
};

void updateLayer(Layer pre, Layer& lay)
{
    for (int i = 0; i < lay.size; i++)
    {
        lay.values[i] = lay.biases[i];
        for (int j = 0; j < pre.size; j++)
        {
            lay.values[i] += pre.values[j] * lay.weights[i][j];
        }

        // faire activation
        // tanh
        lay.values[i] = std::tanh(lay.values[i]);
    }
}

Layer runNN(NeuralNet NN)
{
    for (int l = 1; l < NN.layers.size(); l++)
    {
        updateLayer(NN.layers[l - 1], NN.layers[l]);
    }

    return NN.layers[NN.layers.size() - 1];
}
class Checkpoint
{
public:
    sf::RectangleShape rect;
    sf::Color color = sf::Color::Red;
    int numero;

    std::array<sf::Vector2f, 4> pointsPosition;

    void init(sf::Vector2f center, sf::Vector2f size, float orientation, int num);
};
void Checkpoint::init(sf::Vector2f center, sf::Vector2f size, float orientation, int num)
{
    // create rect
    rect.setSize(size);
    rect.setOrigin(sf::Vector2f(size.x / 2, size.y / 2));
    rect.setRotation(orientation * 57.29);
    rect.setPosition(center);
    rect.setFillColor(color);

    // checkpoint's index
    numero = num;

    // fill points position
    sf::Vector2f v1 = vectAngle(orientation);
    sf::Vector2f v2 = sf::Vector2f(v1.y * -1, v1.x); // v1 + pi/2

    // scale the vectors so they can reach the angles of the rect
    v1.x *= size.x / 2;
    v1.y *= size.x / 2;
    v2.x *= size.y / 2;
    v2.y *= size.y / 2;

    sf::Vector2f p1, p2;
    pointsPosition[0] = addVectors2f(addVectors2f(center, v1), v2);
    pointsPosition[1] = subVectors2f(addVectors2f(center, v1), v2);
    pointsPosition[2] = subVectors2f(subVectors2f(center, v1), v2);
    pointsPosition[3] = addVectors2f(subVectors2f(center, v1), v2);
}
bool pointInRect(sf::Vector2f& p1, Checkpoint& c)
{
    // check if p1 is in a checkpoint
    sf::Vector2f a, b, d;
    a = c.pointsPosition[0];
    b = c.pointsPosition[1];
    d = c.pointsPosition[3];
    float bax = b.x - a.x;
    float bay = b.y - a.y;
    float dax = d.x - a.x;
    float day = d.y - a.y;

    if ((p1.x - a.x) * bax + (p1.y - a.y) * bay < 0)
        return false;
    if ((p1.x - b.x) * bax + (p1.y - b.y) * bay > 0)
        return false;
    if ((p1.x - a.x) * dax + (p1.y - a.y) * day < 0)
        return false;
    if ((p1.x - d.x) * dax + (p1.y - d.y) * day > 0)
        return false;

    return true;
}
class Car
{
public:
    sf::Vector2f position;
    sf::Vector2f speedVect;
    float minspeed = 0.4;
    float factspeed; // speed factor for the NN:   maxspeed = minspeed + nn * factspeed
    sf::Vector2f orientation;

    sf::Vector2f size;
    sf::RectangleShape rect;

    int lastCheckpoint;

    NeuralNet NN;
    bool isdead = false;
    float score;
    float life; // so the cars don't stay alive for too long, life is generated by going though a checkpoint
    float friction = 0.9;
    const float maxlife = 70 / (friction * friction);
    const float viewDist = 300;

    void init(sf::Vector2f pos, sf::Vector2f ori);
    void update(sf::Image& Img, std::vector<Checkpoint>& checkpoints, std::vector<Car>& BestCars);
    void draw(sf::RenderWindow& window);
    float distWall(float angle, sf::Image& Img); // angle = difference with orientation
    void killCar(float addScore, std::vector<Car>& BestCars);
};

void Car::init(sf::Vector2f pos, sf::Vector2f ori)
{
    isdead = false;
    score = 0;
    lastCheckpoint = 0;

    position = pos;
    orientation = ori;
    speedVect = sf::Vector2f(0, 0);
    factspeed = 1;
    life = maxlife;

    rect.setFillColor(HSVtoRGB(rand() % 360, 1, 1));
    size = sf::Vector2f(30, 10);
    rect.setSize(size);
    rect.setOrigin(sf::Vector2f(15, 5));
    rect.setRotation(angleVect(orientation) * 57.29);
    rect.setRotation(3.141592 / 2);

    Layer l1, l2, l3, l4, l5;
    l1.init(10, 0);
    l2.init(5, 10);
    l3.init(5, 5);
    l4.init(3, 5);
    l5.init(2, 3);
    l1.randomize();
    l2.randomize();
    l3.randomize();
    l4.randomize();
    l5.randomize();
    NN.layers.push_back(l1);
    NN.layers.push_back(l2);
    NN.layers.push_back(l3);
    NN.layers.push_back(l4);
    NN.layers.push_back(l5);
}

float Car::distWall(float angle, sf::Image& Img)
{
    float a = angleVect(orientation) + angle;

    sf::Vector2f tp;
    sf::Vector2f vect = sf::Vector2f(cos(a), sin(a));
    float dist = 5;
    bool test = false;

    while (dist < viewDist and !test)
    {
        dist *= 1.5;
        tp.x = vect.x * dist + position.x;
        tp.y = vect.y * dist + position.y;
        test = Img.getPixel(fmax(0, fmin(tp.x, 1919)), fmax(0, fmin(tp.y, 1079))) == sf::Color(0, 0, 0);
    }

    return dist;
}
void Car::killCar(float addScore, std::vector<Car>& BestCars)
{
    score += addScore / 3;
    isdead = true;

    for (int i = 0; i < BestCars.size(); i++)
    {
        if (score > BestCars[i].score)
        {
            BestCars[i].NN = NN;
            BestCars[i].score = score;
            return;
        }
    }void Car::update(sf::Image& Img, std::vector<Checkpoint>& checkpoints, std::vector<Car>& BestCars)
{
    if (!isdead && (life <= 0))
    {
        killCar(maxlife / 3, BestCars);
    }

    if (!isdead)
    {
        NN.layers[0].values[0] = distWall(-3.141592 / 3, Img);
        NN.layers[0].values[1] = distWall(-3.141592 / 4, Img);
        NN.layers[0].values[2] = distWall(-3.141592 / 6, Img);
        NN.layers[0].values[3] = distWall(0, Img);
        NN.layers[0].values[4] = distWall(3.141592 / 6, Img);
        NN.layers[0].values[5] = distWall(3.141592 / 4, Img);
        NN.layers[0].values[6] = distWall(3.141592 / 3, Img);
        NN.layers[0].values[7] = angleVect(orientation) / 10;
        NN.layers[0].values[8] = speedVect.x / 5;
        NN.layers[0].values[9] = speedVect.y / 5;



