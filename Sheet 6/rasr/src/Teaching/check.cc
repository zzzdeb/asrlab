#include "SearchInterface.hh"
#include <Core/Application.hh>
#include <Modules.hh>

class TestApplication : public virtual Core::Application {
public:
  TestApplication() : Core::Application() { setTitle("check"); }

  std::string getUsage() const { return ""; }

  int main(const std::vector<std::string> &arguments) { return EXIT_SUCCESS; }
};

APPLICATION(TestApplication);
