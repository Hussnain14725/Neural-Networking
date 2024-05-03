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
