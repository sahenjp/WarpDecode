#include "../src/gguf.hpp"
#include <iostream>
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    try {
        wd::GgufFile g(argv[1]);
        std::cout << g.version() << " " << g.tensors().size() << " " << g.data_offset() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
