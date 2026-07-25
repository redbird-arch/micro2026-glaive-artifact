#include "thread_output.h"

namespace tacos {

thread_local std::ostream* ThreadOutput::current_stream = &std::cout;

void ThreadOutput::setOutputStream(std::ostream& os) {
    current_stream = &os;
}

std::ostream& ThreadOutput::getOutputStream() {
    if (current_stream) {
        return *current_stream;
    }
    return std::cout; // fallback to cout
}

} // namespace tacos