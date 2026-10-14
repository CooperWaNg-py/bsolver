#include "pool/http_client.h"

#include <memory>
#include <mutex>
#include <stdexcept>

#if BSOLVER_HTTPS_CURL
#include <curl/curl.h>
#elif BSOLVER_HTTPS_WINHTTP
#include <windows.h>
#include <winhttp.h>
#endif

namespace bsolver {

namespace {
[[maybe_unused]] const std::string kUserAgent = std::string("bsolver/") + BSOLVER_VERSION;

[[maybe_unused]] void checkHeaders(const HttpRequest& r) {
  for (const auto& [k, v] : r.headers) {
    if (k.find_first_of("\r\n:") != std::string::npos || v.find_first_of("\r\n") != std::string::npos) {
      throw std::invalid_argument("invalid HTTP header '" + k + "'");
    }
  }
}
}  // namespace

#if BSOLVER_HTTPS_CURL

bool httpsSupported() { return true; }

namespace {
size_t appendBody(char* data, size_t size, size_t n, void* user) {
  static_cast<std::string*>(user)->append(data, size * n);
  return size * n;
}
}  // namespace

HttpResponse httpRequest(const HttpRequest& r) {
  checkHeaders(r);
  static std::once_flag init;
  std::call_once(init, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });

  std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(), curl_easy_cleanup);
  if (!curl) throw std::runtime_error("curl_easy_init failed");
  std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> headers(nullptr, curl_slist_free_all);
  for (const auto& [k, v] : r.headers) {
    // Returns the (unchanged) head, or nullptr on allocation failure with the list left intact.
    curl_slist* head = curl_slist_append(headers.get(), (k + ": " + v).c_str());
    if (!head) throw std::runtime_error("HTTP request failed: cannot build request headers");
    if (!headers) headers.reset(head);
  }

  HttpResponse res;
  CURL* c = curl.get();
  curl_easy_setopt(c, CURLOPT_URL, r.url.c_str());
  // Config-supplied URLs: no file://, ftp:// etc. Redirects are not followed (libcurl default).
#if LIBCURL_VERSION_NUM >= 0x075500  // 7.85.0 added CURLOPT_PROTOCOLS_STR and deprecated CURLOPT_PROTOCOLS
  CURLcode protoRc = curl_easy_setopt(c, CURLOPT_PROTOCOLS_STR, "http,https");
#else
  CURLcode protoRc = curl_easy_setopt(c, CURLOPT_PROTOCOLS, static_cast<long>(CURLPROTO_HTTP | CURLPROTO_HTTPS));
#endif
  if (protoRc != CURLE_OK) {
    throw std::runtime_error(std::string("HTTP request failed: ") + curl_easy_strerror(protoRc));
  }
  curl_easy_setopt(c, CURLOPT_USERAGENT, kUserAgent.c_str());
  curl_easy_setopt(c, CURLOPT_HTTPHEADER, headers.get());
  curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(c, CURLOPT_TIMEOUT, static_cast<long>(r.timeout.count()));
  curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, appendBody);
  curl_easy_setopt(c, CURLOPT_WRITEDATA, &res.body);
  if (r.method == "POST") {
    curl_easy_setopt(c, CURLOPT_POST, 1L);
  } else if (r.method != "GET") {
    curl_easy_setopt(c, CURLOPT_CUSTOMREQUEST, r.method.c_str());
  }
  if (r.method != "GET") {
    curl_easy_setopt(c, CURLOPT_POSTFIELDS, r.body.c_str());
    curl_easy_setopt(c, CURLOPT_POSTFIELDSIZE, static_cast<long>(r.body.size()));
  }
  CURLcode rc = curl_easy_perform(c);
  if (rc != CURLE_OK) throw std::runtime_error(std::string("HTTP request failed: ") + curl_easy_strerror(rc));
  curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &res.status);
  return res;
}

#elif BSOLVER_HTTPS_WINHTTP

bool httpsSupported() { return true; }

namespace {
std::wstring widen(const std::string& s) {
  if (s.empty()) return {};
  int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
  std::wstring w(static_cast<size_t>(n), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
  return w;
}

[[noreturn]] void fail(const char* what) {
  throw std::runtime_error(std::string("HTTP request failed: ") + what + " (WinHTTP error " +
                           std::to_string(GetLastError()) + ")");
}

struct HandleCloser {
  void operator()(HINTERNET h) const { WinHttpCloseHandle(h); }
};
using Handle = std::unique_ptr<void, HandleCloser>;
}  // namespace

HttpResponse httpRequest(const HttpRequest& r) {
  checkHeaders(r);
  std::wstring url = widen(r.url);
  URL_COMPONENTS uc{};
  uc.dwStructSize = sizeof(uc);
  uc.dwHostNameLength = static_cast<DWORD>(-1);
  uc.dwUrlPathLength = static_cast<DWORD>(-1);
  uc.dwExtraInfoLength = static_cast<DWORD>(-1);
  if (!WinHttpCrackUrl(url.c_str(), 0, 0, &uc)) fail("invalid URL");
  if (uc.nScheme != INTERNET_SCHEME_HTTP && uc.nScheme != INTERNET_SCHEME_HTTPS) {
    throw std::runtime_error("HTTP request failed: only http:// and https:// URLs are supported");
  }
  std::wstring host(uc.lpszHostName, uc.dwHostNameLength);
  std::wstring object(uc.lpszUrlPath, uc.dwUrlPathLength);
  if (uc.lpszExtraInfo) object.append(uc.lpszExtraInfo, uc.dwExtraInfoLength);
  if (object.empty()) object = L"/";

  Handle session(WinHttpOpen(widen(kUserAgent).c_str(), WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME,
                             WINHTTP_NO_PROXY_BYPASS, 0));
  if (!session) fail("WinHttpOpen");
  int ms = static_cast<int>(r.timeout.count() * 1000);
  WinHttpSetTimeouts(session.get(), ms, ms, ms, ms);
  Handle connection(WinHttpConnect(session.get(), host.c_str(), uc.nPort, 0));
  if (!connection) fail("connect");
  DWORD flags = uc.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0;
  Handle request(WinHttpOpenRequest(connection.get(), widen(r.method).c_str(), object.c_str(), nullptr,
                                    WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, flags));
  if (!request) fail("open request");
  // Match libcurl: never follow redirects (they would resend the UserToken header to the new location).
  DWORD redirectPolicy = WINHTTP_OPTION_REDIRECT_POLICY_NEVER;
  if (!WinHttpSetOption(request.get(), WINHTTP_OPTION_REDIRECT_POLICY, &redirectPolicy, sizeof(redirectPolicy))) {
    fail("disable redirects");
  }

  std::wstring headers;
  for (const auto& [k, v] : r.headers) headers += widen(k + ": " + v + "\r\n");
  void* body = r.body.empty() ? WINHTTP_NO_REQUEST_DATA : const_cast<char*>(r.body.data());
  DWORD bodyLen = static_cast<DWORD>(r.body.size());
  if (!WinHttpSendRequest(request.get(), headers.empty() ? WINHTTP_NO_ADDITIONAL_HEADERS : headers.c_str(),
                          headers.empty() ? 0 : static_cast<DWORD>(-1L), body, bodyLen, bodyLen, 0)) {
    fail("send");
  }
  if (!WinHttpReceiveResponse(request.get(), nullptr)) fail("receive");

  HttpResponse res;
  DWORD status = 0, size = sizeof(status);
  if (!WinHttpQueryHeaders(request.get(), WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                           WINHTTP_HEADER_NAME_BY_INDEX, &status, &size, WINHTTP_NO_HEADER_INDEX)) {
    fail("status");
  }
  res.status = static_cast<long>(status);
  for (;;) {
    DWORD avail = 0;
    if (!WinHttpQueryDataAvailable(request.get(), &avail)) fail("read");
    if (avail == 0) break;
    size_t old = res.body.size();
    res.body.resize(old + avail);
    DWORD got = 0;
    if (!WinHttpReadData(request.get(), res.body.data() + old, avail, &got)) fail("read");
    res.body.resize(old + got);
  }
  return res;
}

#else

bool httpsSupported() { return false; }

HttpResponse httpRequest(const HttpRequest&) {
  throw std::runtime_error("bsolver was built without an HTTPS client (libcurl not found at configure time)");
}

#endif

}  // namespace bsolver
