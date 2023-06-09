// A Bison parser, made by GNU Bison 3.0.4.

// Skeleton implementation for Bison LALR(1) parsers in C++

// Copyright (C) 2002-2015 Free Software Foundation, Inc.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

// As a special exception, you may create a larger work that contains
// part or all of the Bison parser skeleton and distribute that work
// under terms of your choice, so long as that work isn't itself a
// parser generator using the skeleton or a modified version thereof
// as a parser skeleton.  Alternatively, if you modify or redistribute
// the parser skeleton itself, you may (at your option) remove this
// special exception, which will cause the skeleton and the resulting
// Bison output files to be licensed under the GNU General Public
// License without this special exception.

// This special exception was added by the Free Software Foundation in
// version 2.2 of Bison.

// Take the name prefix into account.
#define yylex BisonParserlex

// First part of user declarations.

#line 39 "ArithmeticExpressionParser.cc" // lalr1.cc:404

#ifndef YY_NULLPTR
#if defined __cplusplus && 201103L <= __cplusplus
#define YY_NULLPTR nullptr
#else
#define YY_NULLPTR 0
#endif
#endif

#include "ArithmeticExpressionParser.hh"

// User implementation prologue.
#line 83 "ArithmeticExpressionParser.yy" // lalr1.cc:412

namespace {
#ifndef BISON_VERSION_1
int yylex(BISON_SEMANTIC_TYPE *yylval, BISON_LOCATION_TYPE *yyloc,
          Core::ArithmeticExpressionParserDriver *driver);
#else
int yylex(BISON_SEMANTIC_TYPE *yylval, BISON_LOCATION_TYPE *yyloc);
Core::ArithmeticExpressionParserDriver *driver;
// old bison parsers need global variables :(
#endif
} // namespace

#line 68 "ArithmeticExpressionParser.cc" // lalr1.cc:412

#ifndef YY_
#if defined YYENABLE_NLS && YYENABLE_NLS
#if ENABLE_NLS
#include <libintl.h> // FIXME: INFRINGES ON USER NAME SPACE.
#define YY_(msgid) dgettext("bison-runtime", msgid)
#endif
#endif
#ifndef YY_
#define YY_(msgid) msgid
#endif
#endif

#define YYRHSLOC(Rhs, K) ((Rhs)[K].location)
/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#ifndef YYLLOC_DEFAULT
#define YYLLOC_DEFAULT(Current, Rhs, N)                                        \
  do                                                                           \
    if (N) {                                                                   \
      (Current).begin = YYRHSLOC(Rhs, 1).begin;                                \
      (Current).end = YYRHSLOC(Rhs, N).end;                                    \
    } else {                                                                   \
      (Current).begin = (Current).end = YYRHSLOC(Rhs, 0).end;                  \
    }                                                                          \
  while (/*CONSTCOND*/ false)
#endif

// Suppress unused-variable warnings by "using" E.
#define YYUSE(E) ((void)(E))

// Enable debugging if requested.
#if YYDEBUG

// A pseudo ostream that takes yydebug_ into account.
#define YYCDEBUG                                                               \
  if (yydebug_)                                                                \
  (*yycdebug_)

#define YY_SYMBOL_PRINT(Title, Symbol)                                         \
  do {                                                                         \
    if (yydebug_) {                                                            \
      *yycdebug_ << Title << ' ';                                              \
      yy_print_(*yycdebug_, Symbol);                                           \
      *yycdebug_ << std::endl;                                                 \
    }                                                                          \
  } while (false)

#define YY_REDUCE_PRINT(Rule)                                                  \
  do {                                                                         \
    if (yydebug_)                                                              \
      yy_reduce_print_(Rule);                                                  \
  } while (false)

#define YY_STACK_PRINT()                                                       \
  do {                                                                         \
    if (yydebug_)                                                              \
      yystack_print_();                                                        \
  } while (false)

#else // !YYDEBUG

#define YYCDEBUG                                                               \
  if (false)                                                                   \
  std::cerr
#define YY_SYMBOL_PRINT(Title, Symbol) YYUSE(Symbol)
#define YY_REDUCE_PRINT(Rule) static_cast<void>(0)
#define YY_STACK_PRINT() static_cast<void>(0)

#endif // !YYDEBUG

#define yyerrok (yyerrstatus_ = 0)
#define yyclearin (yyla.clear())

#define YYACCEPT goto yyacceptlab
#define YYABORT goto yyabortlab
#define YYERROR goto yyerrorlab
#define YYRECOVERING() (!!yyerrstatus_)

namespace BisonParser {
#line 154 "ArithmeticExpressionParser.cc" // lalr1.cc:479

/* Return YYSTR after stripping away unnecessary quotes and
   backslashes, so that it's suitable for yyerror.  The heuristic is
   that double-quoting is unnecessary unless the string contains an
   apostrophe, a comma, or backslash (other than backslash-backslash).
   YYSTR is taken from yytname.  */
std::string ArithmeticExpressionParser::yytnamerr_(const char *yystr) {
  if (*yystr == '"') {
    std::string yyr = "";
    char const *yyp = yystr;

    for (;;)
      switch (*++yyp) {
      case '\'':
      case ',':
        goto do_not_strip_quotes;

      case '\\':
        if (*++yyp != '\\')
          goto do_not_strip_quotes;
        // Fall through.
      default:
        yyr += *yyp;
        break;

      case '"':
        return yyr;
      }
  do_not_strip_quotes:;
  }

  return yystr;
}

/// Build a parser object.
ArithmeticExpressionParser::ArithmeticExpressionParser(
    Core::ArithmeticExpressionParserDriver *driver_yyarg)
    :
#if YYDEBUG
      yydebug_(false), yycdebug_(&std::cerr),
#endif
      driver(driver_yyarg) {
}

ArithmeticExpressionParser::~ArithmeticExpressionParser() {}

/*---------------.
| Symbol types.  |
`---------------*/

inline ArithmeticExpressionParser::syntax_error::syntax_error(
    const location_type &l, const std::string &m)
    : std::runtime_error(m), location(l) {}

// basic_symbol.
template <typename Base>
inline ArithmeticExpressionParser::basic_symbol<Base>::basic_symbol()
    : value() {}

template <typename Base>
inline ArithmeticExpressionParser::basic_symbol<Base>::basic_symbol(
    const basic_symbol &other)
    : Base(other), value(), location(other.location) {
  value = other.value;
}

template <typename Base>
inline ArithmeticExpressionParser::basic_symbol<Base>::basic_symbol(
    typename Base::kind_type t, const semantic_type &v, const location_type &l)
    : Base(t), value(v), location(l) {}

/// Constructor for valueless symbols.
template <typename Base>
inline ArithmeticExpressionParser::basic_symbol<Base>::basic_symbol(
    typename Base::kind_type t, const location_type &l)
    : Base(t), value(), location(l) {}

template <typename Base>
inline ArithmeticExpressionParser::basic_symbol<Base>::~basic_symbol() {
  clear();
}

template <typename Base>
inline void ArithmeticExpressionParser::basic_symbol<Base>::clear() {
  Base::clear();
}

template <typename Base>
inline bool ArithmeticExpressionParser::basic_symbol<Base>::empty() const {
  return Base::type_get() == empty_symbol;
}

template <typename Base>
inline void
ArithmeticExpressionParser::basic_symbol<Base>::move(basic_symbol &s) {
  super_type::move(s);
  value = s.value;
  location = s.location;
}

// by_type.
inline ArithmeticExpressionParser::by_type::by_type() : type(empty_symbol) {}

inline ArithmeticExpressionParser::by_type::by_type(const by_type &other)
    : type(other.type) {}

inline ArithmeticExpressionParser::by_type::by_type(token_type t)
    : type(yytranslate_(t)) {}

inline void ArithmeticExpressionParser::by_type::clear() {
  type = empty_symbol;
}

inline void ArithmeticExpressionParser::by_type::move(by_type &that) {
  type = that.type;
  that.clear();
}

inline int ArithmeticExpressionParser::by_type::type_get() const {
  return type;
}

// by_state.
inline ArithmeticExpressionParser::by_state::by_state() : state(empty_state) {}

inline ArithmeticExpressionParser::by_state::by_state(const by_state &other)
    : state(other.state) {}

inline void ArithmeticExpressionParser::by_state::clear() {
  state = empty_state;
}

inline void ArithmeticExpressionParser::by_state::move(by_state &that) {
  state = that.state;
  that.clear();
}

inline ArithmeticExpressionParser::by_state::by_state(state_type s)
    : state(s) {}

inline ArithmeticExpressionParser::symbol_number_type
ArithmeticExpressionParser::by_state::type_get() const {
  if (state == empty_state)
    return empty_symbol;
  else
    return yystos_[state];
}

inline ArithmeticExpressionParser::stack_symbol_type::stack_symbol_type() {}

inline ArithmeticExpressionParser::stack_symbol_type::stack_symbol_type(
    state_type s, symbol_type &that)
    : super_type(s, that.location) {
  value = that.value;
  // that is emptied.
  that.type = empty_symbol;
}

inline ArithmeticExpressionParser::stack_symbol_type &
ArithmeticExpressionParser::stack_symbol_type::operator=(
    const stack_symbol_type &that) {
  state = that.state;
  value = that.value;
  location = that.location;
  return *this;
}

template <typename Base>
inline void
ArithmeticExpressionParser::yy_destroy_(const char *yymsg,
                                        basic_symbol<Base> &yysym) const {
  if (yymsg)
    YY_SYMBOL_PRINT(yymsg, yysym);

  // User destructor.
  YYUSE(yysym.type_get());
}

#if YYDEBUG
template <typename Base>
void ArithmeticExpressionParser::yy_print_(
    std::ostream &yyo, const basic_symbol<Base> &yysym) const {
  std::ostream &yyoutput = yyo;
  YYUSE(yyoutput);
  symbol_number_type yytype = yysym.type_get();
  // Avoid a (spurious) G++ 4.8 warning about "array subscript is
  // below array bounds".
  if (yysym.empty())
    std::abort();
  yyo << (yytype < yyntokens_ ? "token" : "nterm") << ' ' << yytname_[yytype]
      << " (" << yysym.location << ": ";
  switch (yytype) {
  case 4: // "number"

#line 104 "ArithmeticExpressionParser.yy" // lalr1.cc:636
  {
    debug_stream() << (yysym.value.fval);
  }
#line 426 "ArithmeticExpressionParser.cc" // lalr1.cc:636
  break;

  case 5: // "expression"

#line 104 "ArithmeticExpressionParser.yy" // lalr1.cc:636
  {
    debug_stream() << (yysym.value.fval);
  }
#line 433 "ArithmeticExpressionParser.cc" // lalr1.cc:636
  break;

  default:
    break;
  }
  yyo << ')';
}
#endif

inline void ArithmeticExpressionParser::yypush_(const char *m, state_type s,
                                                symbol_type &sym) {
  stack_symbol_type t(s, sym);
  yypush_(m, t);
}

inline void ArithmeticExpressionParser::yypush_(const char *m,
                                                stack_symbol_type &s) {
  if (m)
    YY_SYMBOL_PRINT(m, s);
  yystack_.push(s);
}

inline void ArithmeticExpressionParser::yypop_(unsigned int n) {
  yystack_.pop(n);
}

#if YYDEBUG
std::ostream &ArithmeticExpressionParser::debug_stream() const {
  return *yycdebug_;
}

void ArithmeticExpressionParser::set_debug_stream(std::ostream &o) {
  yycdebug_ = &o;
}

ArithmeticExpressionParser::debug_level_type
ArithmeticExpressionParser::debug_level() const {
  return yydebug_;
}

void ArithmeticExpressionParser::set_debug_level(debug_level_type l) {
  yydebug_ = l;
}
#endif // YYDEBUG

inline ArithmeticExpressionParser::state_type
ArithmeticExpressionParser::yy_lr_goto_state_(state_type yystate, int yysym) {
  int yyr = yypgoto_[yysym - yyntokens_] + yystate;
  if (0 <= yyr && yyr <= yylast_ && yycheck_[yyr] == yystate)
    return yytable_[yyr];
  else
    return yydefgoto_[yysym - yyntokens_];
}

inline bool ArithmeticExpressionParser::yy_pact_value_is_default_(int yyvalue) {
  return yyvalue == yypact_ninf_;
}

inline bool ArithmeticExpressionParser::yy_table_value_is_error_(int yyvalue) {
  return yyvalue == yytable_ninf_;
}

int ArithmeticExpressionParser::parse() {
  // State.
  int yyn;
  /// Length of the RHS of the rule being reduced.
  int yylen = 0;

  // Error handling.
  int yynerrs_ = 0;
  int yyerrstatus_ = 0;

  /// The lookahead symbol.
  symbol_type yyla;

  /// The locations where the error started and ended.
  stack_symbol_type yyerror_range[3];

  /// The return value of parse ().
  int yyresult;

  // FIXME: This shoud be completely indented.  It is not yet to
  // avoid gratuitous conflicts when merging into the master branch.
  try {
    YYCDEBUG << "Starting parse" << std::endl;

    /* Initialize the stack.  The initial state will be set in
       yynewstate, since the latter expects the semantical and the
       location values to have been already stored, initialize these
       stacks with a primary value.  */
    yystack_.clear();
    yypush_(YY_NULLPTR, 0, yyla);

    // A new symbol was pushed on the stack.
  yynewstate:
    YYCDEBUG << "Entering state " << yystack_[0].state << std::endl;

    // Accept?
    if (yystack_[0].state == yyfinal_)
      goto yyacceptlab;

    goto yybackup;

    // Backup.
  yybackup:

    // Try to take a decision without lookahead.
    yyn = yypact_[yystack_[0].state];
    if (yy_pact_value_is_default_(yyn))
      goto yydefault;

    // Read a lookahead token.
    if (yyla.empty()) {
      YYCDEBUG << "Reading a token: ";
      try {
        yyla.type = yytranslate_(yylex(&yyla.value, &yyla.location, driver));
      } catch (const syntax_error &yyexc) {
        error(yyexc);
        goto yyerrlab1;
      }
    }
    YY_SYMBOL_PRINT("Next token is", yyla);

    /* If the proper action on seeing token YYLA.TYPE is to reduce or
       to detect an error, take that action.  */
    yyn += yyla.type_get();
    if (yyn < 0 || yylast_ < yyn || yycheck_[yyn] != yyla.type_get())
      goto yydefault;

    // Reduce or error.
    yyn = yytable_[yyn];
    if (yyn <= 0) {
      if (yy_table_value_is_error_(yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

    // Count tokens shifted since error; after three, turn off error status.
    if (yyerrstatus_)
      --yyerrstatus_;

    // Shift the lookahead token.
    yypush_("Shifting", yyn, yyla);
    goto yynewstate;

  /*-----------------------------------------------------------.
  | yydefault -- do the default action for the current state.  |
  `-----------------------------------------------------------*/
  yydefault:
    yyn = yydefact_[yystack_[0].state];
    if (yyn == 0)
      goto yyerrlab;
    goto yyreduce;

  /*-----------------------------.
  | yyreduce -- Do a reduction.  |
  `-----------------------------*/
  yyreduce:
    yylen = yyr2_[yyn];
    {
      stack_symbol_type yylhs;
      yylhs.state = yy_lr_goto_state_(yystack_[yylen].state, yyr1_[yyn]);
      /* If YYLEN is nonzero, implement the default value of the
         action: '$$ = $1'.  Otherwise, use the top of the stack.

         Otherwise, the following line sets YYLHS.VALUE to garbage.
         This behavior is undocumented and Bison users should not rely
         upon it.  */
      if (yylen)
        yylhs.value = yystack_[yylen - 1].value;
      else
        yylhs.value = yystack_[0].value;

      // Compute the default @$.
      {
        slice<stack_symbol_type, stack_type> slice(yystack_, yylen);
        YYLLOC_DEFAULT(yylhs.location, slice, yylen);
      }

      // Perform the reduction.
      YY_REDUCE_PRINT(yyn);
      try {
        switch (yyn) {
        case 2:
#line 108 "ArithmeticExpressionParser.yy" // lalr1.cc:859
        {
          driver->setResult((yystack_[0].value.fval));
        }
#line 653 "ArithmeticExpressionParser.cc" // lalr1.cc:859
        break;

        case 3:
#line 115 "ArithmeticExpressionParser.yy" // lalr1.cc:859
        {
          (yylhs.value.fval) = (yystack_[0].value.fval);
        }
#line 659 "ArithmeticExpressionParser.cc" // lalr1.cc:859
        break;

        case 4:
#line 116 "ArithmeticExpressionParser.yy" // lalr1.cc:859
        {
          (yylhs.value.fval) =
              (*(yystack_[3].value.func))((yystack_[1].value.fval));
        }
#line 665 "ArithmeticExpressionParser.cc" // lalr1.cc:859
        break;

        case 5:
#line 117 "ArithmeticExpressionParser.yy" // lalr1.cc:859
        {
          (yylhs.value.fval) =
              (yystack_[2].value.fval) + (yystack_[0].value.fval);
        }
#line 671 "ArithmeticExpressionParser.cc" // lalr1.cc:859
        break;

        case 6:
#line 118 "ArithmeticExpressionParser.yy" // lalr1.cc:859
        {
          (yylhs.value.fval) =
              (yystack_[2].value.fval) - (yystack_[0].value.fval);
        }
#line 677 "ArithmeticExpressionParser.cc" // lalr1.cc:859
        break;

        case 7:
#line 119 "ArithmeticExpressionParser.yy" // lalr1.cc:859
        {
          (yylhs.value.fval) =
              (yystack_[2].value.fval) * (yystack_[0].value.fval);
        }
#line 683 "ArithmeticExpressionParser.cc" // lalr1.cc:859
        break;

        case 8:
#line 120 "ArithmeticExpressionParser.yy" // lalr1.cc:859
        {
          (yylhs.value.fval) =
              (yystack_[2].value.fval) / (yystack_[0].value.fval);
        }
#line 689 "ArithmeticExpressionParser.cc" // lalr1.cc:859
        break;

        case 9:
#line 121 "ArithmeticExpressionParser.yy" // lalr1.cc:859
        {
          (yylhs.value.fval) = -(yystack_[0].value.fval);
        }
#line 695 "ArithmeticExpressionParser.cc" // lalr1.cc:859
        break;

        case 10:
#line 122 "ArithmeticExpressionParser.yy" // lalr1.cc:859
        {
          (yylhs.value.fval) =
              std::pow((yystack_[2].value.fval), (yystack_[0].value.fval));
        }
#line 701 "ArithmeticExpressionParser.cc" // lalr1.cc:859
        break;

        case 11:
#line 123 "ArithmeticExpressionParser.yy" // lalr1.cc:859
        {
          (yylhs.value.fval) = (yystack_[1].value.fval);
        }
#line 707 "ArithmeticExpressionParser.cc" // lalr1.cc:859
        break;

#line 711 "ArithmeticExpressionParser.cc" // lalr1.cc:859
        default:
          break;
        }
      } catch (const syntax_error &yyexc) {
        error(yyexc);
        YYERROR;
      }
      YY_SYMBOL_PRINT("-> $$ =", yylhs);
      yypop_(yylen);
      yylen = 0;
      YY_STACK_PRINT();

      // Shift the result of the reduction.
      yypush_(YY_NULLPTR, yylhs);
    }
    goto yynewstate;

  /*--------------------------------------.
  | yyerrlab -- here on detecting error.  |
  `--------------------------------------*/
  yyerrlab:
    // If not already recovering from an error, report this error.
    if (!yyerrstatus_) {
      ++yynerrs_;
      error(yyla.location, yysyntax_error_(yystack_[0].state, yyla));
    }

    yyerror_range[1].location = yyla.location;
    if (yyerrstatus_ == 3) {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      // Return failure if at end of input.
      if (yyla.type_get() == yyeof_)
        YYABORT;
      else if (!yyla.empty()) {
        yy_destroy_("Error: discarding", yyla);
        yyla.clear();
      }
    }

    // Else will try to reuse lookahead token after shifting the error token.
    goto yyerrlab1;

  /*---------------------------------------------------.
  | yyerrorlab -- error raised explicitly by YYERROR.  |
  `---------------------------------------------------*/
  yyerrorlab:

    /* Pacify compilers like GCC when the user code never invokes
       YYERROR and the label yyerrorlab therefore never appears in user
       code.  */
    if (false)
      goto yyerrorlab;
    yyerror_range[1].location = yystack_[yylen - 1].location;
    /* Do not reclaim the symbols of the rule whose action triggered
       this YYERROR.  */
    yypop_(yylen);
    yylen = 0;
    goto yyerrlab1;

  /*-------------------------------------------------------------.
  | yyerrlab1 -- common code for both syntax error and YYERROR.  |
  `-------------------------------------------------------------*/
  yyerrlab1:
    yyerrstatus_ = 3; // Each real token shifted decrements this.
    {
      stack_symbol_type error_token;
      for (;;) {
        yyn = yypact_[yystack_[0].state];
        if (!yy_pact_value_is_default_(yyn)) {
          yyn += yyterror_;
          if (0 <= yyn && yyn <= yylast_ && yycheck_[yyn] == yyterror_) {
            yyn = yytable_[yyn];
            if (0 < yyn)
              break;
          }
        }

        // Pop the current state because it cannot handle the error token.
        if (yystack_.size() == 1)
          YYABORT;

        yyerror_range[1].location = yystack_[0].location;
        yy_destroy_("Error: popping", yystack_[0]);
        yypop_();
        YY_STACK_PRINT();
      }

      yyerror_range[2].location = yyla.location;
      YYLLOC_DEFAULT(error_token.location, yyerror_range, 2);

      // Shift the error token.
      error_token.state = yyn;
      yypush_("Shifting", error_token);
    }
    goto yynewstate;

    // Accept.
  yyacceptlab:
    yyresult = 0;
    goto yyreturn;

    // Abort.
  yyabortlab:
    yyresult = 1;
    goto yyreturn;

  yyreturn:
    if (!yyla.empty())
      yy_destroy_("Cleanup: discarding lookahead", yyla);

    /* Do not reclaim the symbols of the rule whose action triggered
       this YYABORT or YYACCEPT.  */
    yypop_(yylen);
    while (1 < yystack_.size()) {
      yy_destroy_("Cleanup: popping", yystack_[0]);
      yypop_();
    }

    return yyresult;
  } catch (...) {
    YYCDEBUG << "Exception caught: cleaning lookahead and stack" << std::endl;
    // Do not try to display the values of the reclaimed symbols,
    // as their printer might throw an exception.
    if (!yyla.empty())
      yy_destroy_(YY_NULLPTR, yyla);

    while (1 < yystack_.size()) {
      yy_destroy_(YY_NULLPTR, yystack_[0]);
      yypop_();
    }
    throw;
  }
}

void ArithmeticExpressionParser::error(const syntax_error &yyexc) {
  error(yyexc.location, yyexc.what());
}

// Generate an error message.
std::string
ArithmeticExpressionParser::yysyntax_error_(state_type yystate,
                                            const symbol_type &yyla) const {
  // Number of reported tokens (one for the "unexpected", one per
  // "expected").
  size_t yycount = 0;
  // Its maximum.
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  // Arguments of yyformat.
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];

  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yyla) is
       if this state is a consistent state with a default action.
       Thus, detecting the absence of a lookahead is sufficient to
       determine that there is no unexpected or expected token to
       report.  In that case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is
       a consistent state with a default action.  There might have
       been a previous inconsistent state, consistent state with a
       non-default action, or user semantic action that manipulated
       yyla.  (However, yyla is currently not documented for users.)
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state
       merging (from LALR or IELR) and default reductions corrupt the
       expected token list.  However, the list is correct for
       canonical LR with one exception: it will still contain any
       token that will not be accepted due to an error action in a
       later state.
  */
  if (!yyla.empty()) {
    int yytoken = yyla.type_get();
    yyarg[yycount++] = yytname_[yytoken];
    int yyn = yypact_[yystate];
    if (!yy_pact_value_is_default_(yyn)) {
      /* Start YYX at -YYN if negative to avoid negative indexes in
         YYCHECK.  In other words, skip the first -YYN actions for
         this state because they are default actions.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;
      // Stay within bounds of both yycheck and yytname.
      int yychecklim = yylast_ - yyn + 1;
      int yyxend = yychecklim < yyntokens_ ? yychecklim : yyntokens_;
      for (int yyx = yyxbegin; yyx < yyxend; ++yyx)
        if (yycheck_[yyx + yyn] == yyx && yyx != yyterror_ &&
            !yy_table_value_is_error_(yytable_[yyx + yyn])) {
          if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM) {
            yycount = 1;
            break;
          } else
            yyarg[yycount++] = yytname_[yyx];
        }
    }
  }

  char const *yyformat = YY_NULLPTR;
  switch (yycount) {
#define YYCASE_(N, S)                                                          \
  case N:                                                                      \
    yyformat = S;                                                              \
    break
    YYCASE_(0, YY_("syntax error"));
    YYCASE_(1, YY_("syntax error, unexpected %s"));
    YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
    YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
    YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
    YYCASE_(5,
            YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
#undef YYCASE_
  }

  std::string yyres;
  // Argument number.
  size_t yyi = 0;
  for (char const *yyp = yyformat; *yyp; ++yyp)
    if (yyp[0] == '%' && yyp[1] == 's' && yyi < yycount) {
      yyres += yytnamerr_(yyarg[yyi++]);
      ++yyp;
    } else
      yyres += *yyp;
  return yyres;
}

const signed char ArithmeticExpressionParser::yypact_ninf_ = -7;

const signed char ArithmeticExpressionParser::yytable_ninf_ = -1;

const signed char ArithmeticExpressionParser::yypact_[] = {
    10, -5, -7, 10, 10, 6,  28, 10, 4, 12, -7, 10,
    10, 10, 10, 10, 20, -7, -6, -6, 4, 4,  4,  -7};

const unsigned char ArithmeticExpressionParser::yydefact_[] = {
    0, 0, 3, 0, 0, 0, 2, 0, 9, 0, 1, 0, 0, 0, 0, 0, 0, 11, 5, 6, 7, 8, 10, 4};

const signed char ArithmeticExpressionParser::yypgoto_[] = {-7, -7, -3};

const signed char ArithmeticExpressionParser::yydefgoto_[] = {-1, 5, 6};

const unsigned char ArithmeticExpressionParser::yytable_[] = {
    8,  9,  13, 14, 16, 15, 10, 7,  18, 19, 20, 21, 22, 1,
    2,  15, 0,  3,  11, 12, 13, 14, 4,  15, 0,  17, 11, 12,
    13, 14, 0,  15, 0,  23, 11, 12, 13, 14, 0,  15};

const signed char ArithmeticExpressionParser::yycheck_[] = {
    3, 4, 8,  9,  7,  11, 0, 12, 11, 12, 13, 14, 15, 3,  4, 11, -1, 7, 6,  7,
    8, 9, 12, 11, -1, 13, 6, 7,  8,  9,  -1, 11, -1, 13, 6, 7,  8,  9, -1, 11};

const unsigned char ArithmeticExpressionParser::yystos_[] = {
    0, 3, 4, 7,  12, 15, 16, 12, 16, 16, 0,  6,
    7, 8, 9, 11, 16, 13, 16, 16, 16, 16, 16, 13};

const unsigned char ArithmeticExpressionParser::yyr1_[] = {
    0, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16};

const unsigned char ArithmeticExpressionParser::yyr2_[] = {0, 2, 1, 1, 4, 3,
                                                           3, 3, 3, 2, 3, 3};

// YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
// First, the terminals, then, starting at \a yyntokens_, nonterminals.
const char *const ArithmeticExpressionParser::yytname_[] = {"\"end of string\"",
                                                            "error",
                                                            "$undefined",
                                                            "\"function\"",
                                                            "\"number\"",
                                                            "\"expression\"",
                                                            "'+'",
                                                            "'-'",
                                                            "'*'",
                                                            "'/'",
                                                            "NEG",
                                                            "'^'",
                                                            "'('",
                                                            "')'",
                                                            "$accept",
                                                            "unit",
                                                            "exp",
                                                            YY_NULLPTR};

#if YYDEBUG
const unsigned char ArithmeticExpressionParser::yyrline_[] = {
    0, 108, 108, 115, 116, 117, 118, 119, 120, 121, 122, 123};

// Print the state stack on the debug stream.
void ArithmeticExpressionParser::yystack_print_() {
  *yycdebug_ << "Stack now";
  for (stack_type::const_iterator i = yystack_.begin(), i_end = yystack_.end();
       i != i_end; ++i)
    *yycdebug_ << ' ' << i->state;
  *yycdebug_ << std::endl;
}

// Report on the debug stream that the rule \a yyrule is going to be reduced.
void ArithmeticExpressionParser::yy_reduce_print_(int yyrule) {
  unsigned int yylno = yyrline_[yyrule];
  int yynrhs = yyr2_[yyrule];
  // Print the symbols being reduced, and their result.
  *yycdebug_ << "Reducing stack by rule " << yyrule - 1 << " (line " << yylno
             << "):" << std::endl;
  // The symbols being reduced.
  for (int yyi = 0; yyi < yynrhs; yyi++)
    YY_SYMBOL_PRINT("   $" << yyi + 1 << " =", yystack_[(yynrhs) - (yyi + 1)]);
}
#endif // YYDEBUG

// Symbol number corresponding to token number t.
inline ArithmeticExpressionParser::token_number_type
ArithmeticExpressionParser::yytranslate_(int t) {
  static const token_number_type translate_table[] = {
      0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  2,  2, 2, 2, 2, 2,  2,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 12, 13, 8, 6, 2, 7, 2,  9,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  2,  2, 2, 2, 2, 2,  2,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  2,  2, 2, 2, 2, 11, 2,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  2,  2, 2, 2, 2, 2,  2,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  2,  2, 2, 2, 2, 2,  2,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  2,  2, 2, 2, 2, 2,  2,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  2,  2, 2, 2, 2, 2,  2,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  2,  2, 2, 2, 2, 2,  2,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  2,  2, 2, 2, 2, 2,  2,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1,  2,  3, 4, 5, 10};
  const unsigned int user_token_number_max_ = 261;
  const token_number_type undef_token_ = 2;

  if (static_cast<int>(t) <= yyeof_)
    return yyeof_;
  else if (static_cast<unsigned int>(t) <= user_token_number_max_)
    return translate_table[t];
  else
    return undef_token_;
}

} // namespace BisonParser
#line 1137 "ArithmeticExpressionParser.cc" // lalr1.cc:1167
#line 125 "ArithmeticExpressionParser.yy"  // lalr1.cc:1168

// ========================================================================

#ifndef BISON_VERSION_1
void BISON_NAMESPACE::ArithmeticExpressionParser::error(
    const BISON_LOCATION_TYPE &loc, const std::string &msg) {
  driver->error(loc, msg);
}
#else
/* Bison 1.xx declares these functions but never defines them */
void BISON_NAMESPACE::ArithmeticExpressionParser::error_() {}
void BISON_NAMESPACE::ArithmeticExpressionParser::print_() {}
#endif

// ========================================================================

#include "StringUtilities.hh"
#include <cstdio>
using namespace Core;

ArithmeticExpressionParserDriver::ArithmeticExpressionParserDriver() {
  initializeFunctions();
}

void ArithmeticExpressionParserDriver::initializeFunctions() {
  functions_["log"] = std::log;
  functions_["exp"] = std::exp;
  functions_["sin"] = std::sin;
  functions_["cos"] = std::cos;
  functions_["sqrt"] = std::sqrt;
}

bool ArithmeticExpressionParserDriver::parse(const std::string &input,
                                             double &result) {
  input_ = new LexerInput(input);
#ifndef BISON_VERSION_1
  BISON_NAMESPACE::ArithmeticExpressionParser parser(this);
#else
  BISON_LOCATION_TYPE l;
  driver = this;
  BISON_NAMESPACE::ArithmeticExpressionParser parser(false, l, this);
#endif
  bool error(parser.parse() != 0);
  delete input_;
  result = result_;
  return !error;
}

ArithmeticExpressionParserDriver::LexerInput *
ArithmeticExpressionParserDriver::getLexerInput() const {
  return input_;
}

void ArithmeticExpressionParserDriver::setResult(double r) { result_ = r; }

void ArithmeticExpressionParserDriver::error(
    const BISON_NAMESPACE::BISON_LOCATION &loc, const std::string &msg) {
  lastError_ = form("%d-%d: %s", loc.begin.column, loc.end.column, msg.c_str());
}

std::string ArithmeticExpressionParserDriver::getLastError() const {
  return lastError_;
}

ArithmeticExpressionParserDriver::MathFunc
ArithmeticExpressionParserDriver::getFunction(const std::string &function) {
  std::map<std::string, MathFunc>::const_iterator i;
  i = functions_.find(function);
  if (i == functions_.end())
    return 0;
  else
    return i->second;
}

// ========================================================================

ArithmeticExpressionParserDriver::LexerInput::LexerInput(const std::string &s)
    : str_(s), pos_(0) {}

char ArithmeticExpressionParserDriver::LexerInput::get() const {
  return (pos_ < str_.size() ? str_[pos_] : '\0');
}

const char *ArithmeticExpressionParserDriver::LexerInput::getString() const {
  return (pos_ < str_.size() ? str_.c_str() + pos_ : 0);
}

ArithmeticExpressionParserDriver::LexerInput &
ArithmeticExpressionParserDriver::LexerInput::operator++() {
  ++pos_;
  return *this;
}

ArithmeticExpressionParserDriver::LexerInput &
ArithmeticExpressionParserDriver::LexerInput::operator+=(int p) {
  pos_ += p;
  return *this;
}

// ========================================================================

namespace {
#ifndef BISON_VERSION_1
int yylex(BISON_SEMANTIC_TYPE *yylval, BISON_LOCATION_TYPE *yyloc,
          ArithmeticExpressionParserDriver *driver)
#else
int yylex(BISON_SEMANTIC_TYPE *yylval, BISON_LOCATION_TYPE *yyloc)
#endif
{
  typedef BISON_TOKEN_TYPE token;
  char c;
  ArithmeticExpressionParserDriver::LexerInput *input = driver->getLexerInput();

  while ((c = input->get()) == ' ' || c == '\t') {
    ++*input;
    yyloc->columns(1);
  }
  yyloc->step();
  if (c == '\0') {
    return BISON_TOKEN(END);
  }
  if (c == '.' || isdigit(c)) {
    int read;
    sscanf(input->getString(), "%lf%n", &yylval->fval, &read);
    *input += read;
    yyloc->columns(read);
    return BISON_TOKEN(NUMBER);
  }
  if (isalpha(c)) {
    std::string buffer;
    do {
      buffer += input->get();
      ++*input;
    } while (isalnum(input->get()));
    yyloc->columns(buffer.size());
    yylval->func = driver->getFunction(buffer);
    if (yylval->func == 0) {
      driver->error(*yyloc, std::string("unknown function: ") + buffer);
      yylval->fval = 0;
      return BISON_TOKEN(NUMBER);
    }
    return BISON_TOKEN(FUNCTION);
  }
  ++*input;
  yyloc->columns(1);
  return static_cast<BISON_TOKEN_TYPE>(c);
}
} // namespace
