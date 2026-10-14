#pragma once

#include <chrono>
#include <string>
#include <utility>
#include <vector>

namespace bsolver {

struct HttpRequest {
  std::string method = "GET";
  std::string url;  // http:// or https://
  std::vector<std::pair<std::string, std::string>> headers;
  std::string body;
  std::chrono::seconds timeout{30};
};

struct HttpResponse {
  long status = 0;
  std::string body;
};

// True when bsolver was built with an HTTPS client (libcurl, or WinHTTP on Windows).
bool httpsSupported();

// Performs one request. Throws std::runtime_error on transport failures (DNS, TLS, timeout, ...)
// and std::invalid_argument for header values containing CR/LF. Any HTTP status is returned.
HttpResponse httpRequest(const HttpRequest& request);

}  // namespace bsolver
