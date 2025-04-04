#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <variant>
#include <functional>
#include "icicle/utils/utils.h"

/*
 * DeviceInitializer
 *
 * This class template, `DeviceInitializer<T>`, provides a mechanism for managing
 * the initialization and retrieval of objects of type `T` across multiple devices
 * and contexts. It supports initializing data for different device IDs (represented
 * by `unsigned int`) and context values (using `std::variant` for flexibility in key types).
 *
 * Key Features:
 *  - Singleton Pattern: `DeviceInitializer` is designed as a singleton, providing
 *    a single instance across the application for each type `T`. This ensures that
 *    device and context data for `T` is globally accessible.
 *
 *  - Thread-Safe Access: A `std::mutex` is used to ensure thread-safe operations
 *    on the stored data, making it suitable for concurrent access.
 *
 *  - Contextual Initialization: `DeviceInitializer` allows unique instances of
 *    `T` for each `(device_id, context)` pair, where `device_id` is an `unsigned int`
 *    and `context` is a `std::variant` (capable of storing `int` or `std::string` values).
 *
 * Usage:
 * - `get_or_init(device_id, context, initializer)`: Retrieves the object associated with
 *    the given `device_id` and `context`. If no object exists, it initializes one using
 *    the `initializer` function (defaulting to the default constructor of `T`).
 *
 * - `initialize(device_id, context, value)`: Manually initializes and stores a given
 *    `value` of type `T` for the specified `device_id` and `context`.
 *
 * - `get(device_id, context)`: Retrieves the object for the specified `device_id` and `context`
 *    without initializing it if it does not already exist. Returns a `std::optional` to indicate
 *    the presence of an initialized object.
 *
 */

// Define Context as std::variant for type-safe, comparable keys
using Context = std::variant<unsigned, std::string>;

template <typename T>
class DeviceInitializer
{
public:
  static DeviceInitializer& instance()
  {
    static DeviceInitializer instance;
    return instance;
  }

  // Retrieve or initialize an object for a given device and context
  T& get_or_init(unsigned int device_id, const Context& context, std::function<T()> initializer = [] { return T(); })
  {
    // First, check without locking to avoid unnecessary locking if data is already initialized.
    auto device_iter = m_data.find(device_id);
    if (device_iter != m_data.end()) {
      auto context_iter = device_iter->second.find(context);
      if (context_iter != device_iter->second.end()) {
        // Data is already initialized, return it directly.
        return *context_iter->second;
      }
    }

    // Lock and perform a double-checked lookup
    std::lock_guard<std::mutex> lock(m_mutex);
    auto& device_data = m_data[device_id];
    auto it = device_data.find(context);

    if (it == device_data.end()) {
      // Initialize only if data is still missing after locking
      device_data[context] = std::make_unique<T>(initializer());
    }

    return *device_data[context];
  }

  // Manual initialization method for a given device and context
  void initialize(unsigned int device_id, const Context& context, T value)
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_data[device_id][context] = std::make_unique<T>(std::move(value));
  }

private:
  DeviceInitializer() = default;

  // Map to hold data per device (unsigned int) and per context
  std::map<unsigned int, std::map<Context, std::unique_ptr<T>>> m_data;
  std::mutex m_mutex;
};
