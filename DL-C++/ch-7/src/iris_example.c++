/*  Iris‑classification demo
 *  – single Dense layer + Softmax – Eigen Tensor API (C++17)
 *  – robust CSV loader that accepts both Kaggle & UCI formats
 */

#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <tuple>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <ctime>
#include <unsupported/Eigen/CXX11/Tensor>

/*──────────────── Activation + Softmax ───────────────*/
template <int R>
struct Activation {
    virtual ~Activation() = default;
    virtual Eigen::Tensor<float, R>
    forward(const Eigen::Tensor<float, R> &) const = 0;
};
template<int Rank>
class Softmax : public Activation<Rank> {
public:
    Eigen::Tensor<float, Rank>
    forward(const Eigen::Tensor<float, Rank>& x) const override {

        Eigen::array<int,1> rd{{Rank - 1}};
        auto maxes = x.maximum(rd);

        Eigen::array<Eigen::Index, Rank> rs, bc;
        for (int i = 0; i < Rank-1; ++i) { rs[i]=x.dimension(i); bc[i]=1; }
        rs[Rank-1] = 1;
        bc[Rank-1] = x.dimension(Rank-1);

        auto max_b  = maxes.reshape(rs).broadcast(bc);
        auto exp_x  = (x - max_b).exp();
        auto denom  = exp_x.sum(rd).reshape(rs).broadcast(bc);

        /* --------- critical line --------- */
        return (exp_x / denom).eval();   // owns its memory
    }
};


/*──────────────────────── Dense ──────────────────────*/
class Dense
{
    Eigen::Tensor<float, 2> W_, b_;
    std::unique_ptr<Activation<2>> act_;
    Eigen::Tensor<float, 2> x_cache_, gW_, gb_;

public:
    Dense(Eigen::Tensor<float, 2> W, Eigen::Tensor<float, 2> b,
          std::unique_ptr<Activation<2>> a)
        : W_(std::move(W)), b_(std::move(b)), act_(std::move(a)) {}

    Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 2> &x)
    {
        x_cache_ = x;
        Eigen::array<Eigen::IndexPair<int>, 1> cd{{{1, 0}}};
        Eigen::Tensor<float, 2> z = x.contract(W_, cd).eval();
        Eigen::array<Eigen::Index, 2> bc{{x.dimension(0), 1}};
        z = (z + b_.broadcast(bc)).eval();
        return act_->forward(z);
    }
    Eigen::Tensor<float, 2> predict(const Eigen::Tensor<float, 2> &x) const
    {
        Eigen::array<Eigen::IndexPair<int>, 1> cd{{{1, 0}}};
        Eigen::Tensor<float, 2> z = x.contract(W_, cd).eval();
        Eigen::array<Eigen::Index, 2> bc{{x.dimension(0), 1}};
        z = (z + b_.broadcast(bc)).eval();
        return act_->forward(z);
    }
    void backward(const Eigen::Tensor<float, 2> &dY)
    {
        Eigen::Tensor<float, 2> xT = x_cache_.shuffle(Eigen::array<int, 2>{{1, 0}});
        Eigen::array<Eigen::IndexPair<int>, 1> cd{{{1, 0}}};
        gW_ = xT.contract(dY, cd).eval();
        Eigen::Tensor<float, 1> tmp = dY.sum(Eigen::array<int, 1>{{0}});
        gb_ = tmp.reshape(Eigen::DSizes<Eigen::Index, 2>{1, dY.dimension(1)}).eval();
    }
    void update(float lr)
    {
        W_ -= lr * gW_;
        b_ -=
            lr * gb_;
    }
};

/*────────────────── CCE loss ──────────────────*/
template <int R>
class CategoricalCrossEntropy
{
public:
    float evaluate(const Eigen::Tensor<float, R> &y,
                   const Eigen::Tensor<float, R> &p) const
    {
        auto logp = p.cwiseMax(1e-7f).log();
        /* ▼ evaluate() fix */
        Eigen::Tensor<float, 0> tot =
            (y * logp).sum(Eigen::array<int, 2>{{0, 1}}).eval();
        return -tot() / static_cast<float>(y.dimension(0));
    }
    Eigen::Tensor<float, R> derivative(const Eigen::Tensor<float, R> &y,
                                       const Eigen::Tensor<float, R> &p) const
    {
        return (p - y) / static_cast<float>(y.dimension(0));
    }
};

/*──────────── Robust CSV loader (Kaggle / UCI) ───────────*/
auto load_iris_dataset(const std::string &path, bool shuffle = true, float split = .8f)
    -> std::tuple<Eigen::Tensor<float, 2>, Eigen::Tensor<float, 2>,
                  Eigen::Tensor<float, 2>, Eigen::Tensor<float, 2>>
{
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("open " + path);
    const int N = 150;
    std::vector<std::string> lines;
    lines.reserve(N);
    std::string ln;
    while (std::getline(f, ln))
        lines.push_back(ln);
    if (shuffle)
    {
        std::mt19937 g(std::random_device{}());
        std::shuffle(lines.begin(), lines.end(), g);
    }
    // ---------------------------------------------------------------------
    // 1.  Temporary contiguous buffer
    // ---------------------------------------------------------------------
    std::vector<float> flat; // will hold 150 × 7 floats
    flat.reserve(150 * 7);   // avoid reallocations

    // ---------------------------------------------------------------------
    // 2.  Parse every CSV line → push_back() floats
    // ---------------------------------------------------------------------
    for (const std::string &line : lines)
    {
        if (line.empty())
            continue;

        // split on ',' into tokens
        std::vector<std::string> toks;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ','))
            toks.push_back(tok);

        // skip header rows
        if (!toks.empty() &&
            (toks[0] == "Id" || toks[0] == "SepalLengthCm" ||
             toks[0] == "sepal_length"))
            continue;

        if (toks.size() == 6) /*  Kaggle format:  Id + 4 + Species  */
        {
            for (int i = 1; i <= 4; ++i)
                flat.push_back(std::stof(toks[i])); // 4 flower features

            const std::string &sp = toks[5]; // Species column
            if (sp == "Iris-setosa")
                flat.insert(flat.end(), {1, 0, 0});
            else if (sp == "Iris-versicolor")
                flat.insert(flat.end(), {0, 1, 0});
            else if (sp == "Iris-virginica")
                flat.insert(flat.end(), {0, 0, 1});
            else
                throw std::runtime_error("Unknown class: " + sp);
        }
        else if (toks.size() == 5) /*  UCI format:  4 + Species          */
        {
            for (int i = 0; i < 4; ++i)
                flat.push_back(std::stof(toks[i]));

            const std::string &sp = toks[4];
            if (sp == "Iris-setosa")
                flat.insert(flat.end(), {1, 0, 0});
            else if (sp == "Iris-versicolor")
                flat.insert(flat.end(), {0, 1, 0});
            else if (sp == "Iris-virginica")
                flat.insert(flat.end(), {0, 0, 1});
            else
                throw std::runtime_error("Unknown class: " + sp);
        }
        else
            throw std::runtime_error("Bad CSV line: " + line);
    }

    // ---------------------------------------------------------------------
    // 3.  Wrap that buffer in an Eigen::TensorMap   (150 rows × 7 cols)
    // ---------------------------------------------------------------------
    Eigen::TensorMap<Eigen::Tensor<float, 2>> full(flat.data(), 150, 7);

    // ---------------------------------------------------------------------
    // 4.  Slice into train/test tensors and materialise them with .eval()
    // ---------------------------------------------------------------------
    int n_train = static_cast<int>(150 * split);
    Eigen::DSizes<Eigen::Index, 2> o, s;

    o = {0, 0};
    s = {n_train, 4};
    Eigen::Tensor<float, 2> train_X = full.slice(o, s).eval(); // ★ owns data

    o = {0, 4};
    s = {n_train, 3};
    Eigen::Tensor<float, 2> train_Y = full.slice(o, s).eval(); // ★

    o = {n_train, 0};
    s = {150 - n_train, 4};
    Eigen::Tensor<float, 2> test_X = full.slice(o, s).eval(); // ★

    o = {n_train, 4};
    s = {150 - n_train, 3};
    Eigen::Tensor<float, 2> test_Y = full.slice(o, s).eval(); // ★

    return {train_X, train_Y, test_X, test_Y};
}

/*──────────────────── accuracy ───────────────────*/
float accuracy(const Eigen::Tensor<float, 2> &y,
               const Eigen::Tensor<float, 2> &p)
{
    auto a = y.argmax(1), b = p.argmax(1);
    auto cmp = a.binaryExpr(b, [](auto i, auto j)
                            { return i == j ? 1.f : 0.f; });
    /* ▼ accuracy fix */
    Eigen::Tensor<float, 0> m = cmp.mean().eval();
    return m() * 100.f;
}

/*──────────────────────── main ───────────────────*/
int main(int argc, char **argv)
{
    std::string csv = (argc >= 2) ? argv[1] : "data/Iris.csv";
    if (argc < 2)
        std::cout << "Using default path " << csv << '\n';
    try
    {
        auto [Xtr, Ytr, Xte, Yte] = load_iris_dataset(csv, true, 0.8f);
        auto randT = [](int r, int c, float rng)
        { Eigen::Tensor<float,2> T(r,c);
             T=T.random()*T.constant(rng)-T.constant(rng/2.f); return T; };
        Dense net(randT(4, 3, 0.1f),
                  Eigen::Tensor<float, 2>(1, 3).setZero(),
                  std::make_unique<Softmax<2>>());
        CategoricalCrossEntropy<2> loss;
        const int EPOCHS = 1000;
        const float LR = 0.1f;
        for (int e = 1; e <= EPOCHS; ++e)
        {
            auto out = net.forward(Xtr);
            net.backward(loss.derivative(Ytr, out));
            net.update(LR);
            if (e == 1 || e % 50 == 0)
            {
                float l = loss.evaluate(Ytr, out);
                float a = accuracy(Ytr, out);
                auto vo = net.predict(Xte);
                std::cout << "Epoch " << e << " | loss " << l << " acc " << a
                          << " | val_loss " << loss.evaluate(Yte, vo)
                          << " val_acc " << accuracy(Yte, vo) << '\n';
            }
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Error: " << ex.what() << "\nUsage: " << argv[0] << " [Iris.csv]\n";
        return 1;
    }
    return 0;
}
