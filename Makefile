# File: Makefile

# Compiler and flags
CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors

# Target executable
TARGET = symnmf

# Source and header files
SOURCES = symnmf.c
HEADERS = symnmf.h

# Object files
OBJECTS = $(SOURCES:.c=.o)

# Build the target executable
$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS)

# Compile source files into object files
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(TARGET) $(OBJECTS)
