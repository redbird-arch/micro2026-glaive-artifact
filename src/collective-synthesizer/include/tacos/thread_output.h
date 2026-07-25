/*
# File name  :    thread_output.h
# Author     :    Galois
# Time       :    2025/11/17 15:30:32
*/

#pragma once
#include <iostream>
#include <sstream>

namespace tacos {

class ThreadOutput {
private:
    static thread_local std::ostream* current_stream;

public:
    static void setOutputStream(std::ostream& os);
    
    static std::ostream& getOutputStream();
    
    // Helper function to format and output
    template<typename T>
    static void output(const T& value) {
        getOutputStream() << value;
    }
    
    // Overload for std::endl and other manipulators
    static void output(std::ostream& (*manip)(std::ostream&)) {
        getOutputStream() << manip;
    }
};

} // namespace tacos