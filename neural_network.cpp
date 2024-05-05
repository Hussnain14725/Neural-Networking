#include <iostream>
#include <vector>
#include <thread>
#include <array>
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