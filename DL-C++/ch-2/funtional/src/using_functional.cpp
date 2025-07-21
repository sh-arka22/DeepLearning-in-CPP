#include <algorithm> //std::for_each
#include <functional> //std::less, std::less_
#include <iostream> //std::cout

int main(){
    std::vector<std::function<bool(double, double)>> comparators{
        std::less<double>(), // Using std::less for less than comparison
        std::greater<double>(), // Using std::greater for greater than comparison
        std::less_equal<double>(), // Using std::less_equal for less than or equal to comparison
        std::greater_equal<double>(), // Using std::greater_equal for greater than or equal to comparison
    };

    double x = 10.0, y = 10.0;

    auto compare = [&x, &y]
    (const std::function<bool(double, double)>& comparator) {
        bool result = comparator(x, y);
        std::cout << "Comparison result: " << (result ? "true" : "false") << std::endl;

    };

    std::for_each(comparators.begin(), comparators.end(), compare);
    return 0;
}