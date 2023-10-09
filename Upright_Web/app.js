const mysql = require("mysql2");
const connection = mysql.createConnection({
  host: "210.125.31.236",
  port: "3306",
  user: "test2",
  password: "s1234",
  database: "upright",
});
var id;
connection.connect();

// node_modules 에 있는 express 관련 파일을 가져온다.
var express = require("express");
const { send } = require("express/lib/response");
const { response } = require("express");

// express 는 함수이므로, 반환값을 변수에 저장한다.
var app = express();

// 3000 포트로 서버 오픈
app.listen(3000, function () {
  console.log("start! express server on port 3000");
});

// 이제 터미널에 node app.js 를 입력해보자.

// request 와 response 라는 인자를 줘서 콜백 함수를 만든다.
// localhost:3000 브라우저에 res.send() 내부의 문자열이 띄워진다.

app.get("/", function (req, res) {
  res.sendFile(__dirname + "/public/index.html");
  connection.query("SELECT id FROM upright.now_id", (error, rows, fields) => {
    if (error) throw error;
    console.log("User info is: ", rows[0].id);
    id = rows[0].id;
  });
});

// localhost:3000/main 브라우저에 res.sendFile() 내부의 파일이 띄워진다.
app.get("/main", function (req, res) {
  res.sendFile(__dirname + "/public/generic.html");
});

app.use(express.static("public"));

app.post("/", function (req, res) {
  var responseData = {};
  responseData.score = [];
  let myTutle;
  let myEye;

  connection.query(
    "SELECT turtle, eye, timeCnt FROM upright.habit_count where id='" +
      id +
      "'",
    function (err, rows, fields) {
      //console.log(rows);
      if (err) throw err;
      if (rows[0]) {
        responseData.result = "ok";
        rows.forEach(function (val) {
          myTutle = val.turtle / (val.timeCnt / 600);
          myEye = val.eye / (val.timeCnt / 600);
          responseData.score.push(myTutle);
          responseData.score.push(myEye);
        });
      } else {
        responseData.result = "none";
        responseData.score = "";
      }
    }
  );
  connection.query(
    "SELECT AVG(turtle)as turtle, AVG(eye)as eye, AVG(timeCnt)as timeCnt FROM upright.habit_count",
    function (err, rows, fields) {
      //console.log(rows);
      if (err) throw err;
      if (rows[0]) {
        responseData.result = "ok";
        rows.forEach(function (val) {
          responseData.score.push(myTutle);
          responseData.score.push(val.turtle / (val.timeCnt / 600));
          responseData.score.push(myEye);
          responseData.score.push(val.eye / (val.timeCnt / 600));
        });
      } else {
        responseData.result = "none";
        responseData.score = "";
      }
      res.json(responseData);
    }
  );
});

app.post("/imgCnt", function (req, res) {
  var conn = mysql.createConnection({
    host: "210.125.31.236",
    port: "3306",
    user: "test2",
    password: "s1234",
    database: "upright",
  });
  conn.connect();
  var query = conn.query(
    "select COUNT(img) as cnt from imglist where id='" + id + "' ",
    function (err, rows) {
      if (err) throw err;
      if (rows[0]) {
        rows.forEach(function (val) {
          res.json(val.cnt);
        });
      } else res.json(0);
    }
  );

  app.get("/correctionimg1", function (req, res) {
    var conn = mysql.createConnection({
      host: "210.125.31.236",
      port: "3306",
      user: "test2",
      password: "s1234",
      database: "upright",
    });
    conn.connect();
    var query = conn.query(
      "SELECT img FROM imglist where id='" +
        id +
        "' order by number desc LIMIT 0, 1",
      function (err, rows) {
        if (err) throw err;
        res.type("png");
        if (rows[0]["img"] != null) {
          res.send(rows[0]["img"]);
        }
      }
    );
  });

  // app.get('/correctionimg1', function (req, res) {
  //   var conn = mysql.createConnection({
  //     host: '210.125.31.236',
  //     port: '3306',
  //     user: 'test2',
  //     password: 's1234',
  //     database: 'upright'
  //   })
  //   conn.connect();
  //   var query = conn.query('SELECT img FROM imglist where id=\'' + id + '\' order by number desc', function (err, rows) {
  //     const resData = [];
  //     if (err) throw err;
  //     res.type('png');
  //     for (let i = 0; i < rows.length; i++) {
  //       if (rows[i]['img'] != null) {
  //         resData.push(rows[i]['img'])
  //       }
  //     }
  //     res.send(resData);
  //   });
  // });

  app.get("/correctionimg2", function (req, res) {
    var conn = mysql.createConnection({
      host: "210.125.31.236",
      port: "3306",
      user: "test2",
      password: "s1234",
      database: "upright",
    });
    conn.connect();
    var query = conn.query(
      "SELECT img FROM imglist where id='" +
        id +
        "' order by number desc LIMIT 1, 1",
      function (err, rows) {
        if (err) throw err;
        res.type("png");
        if (rows[0]["img"] != null) res.send(rows[0]["img"]);
      }
    );
  });

  app.get("/correctionimg3", function (req, res) {
    var conn = mysql.createConnection({
      host: "210.125.31.236",
      port: "3306",
      user: "test2",
      password: "s1234",
      database: "upright",
    });
    conn.connect();
    var query = conn.query(
      "SELECT img FROM imglist where id='" +
        id +
        "' order by number desc LIMIT 2, 1",
      function (err, rows) {
        if (err) throw err;
        res.type("png");
        if (rows[0]["img"] != null) res.send(rows[0]["img"]);
      }
    );
  });

  app.get("/correctionimg4", function (req, res) {
    var conn = mysql.createConnection({
      host: "210.125.31.236",
      port: "3306",
      user: "test2",
      password: "s1234",
      database: "upright",
    });
    conn.connect();
    var query = conn.query(
      "SELECT img FROM imglist where id='" +
        id +
        "' order by number desc LIMIT 3, 1",
      function (err, rows) {
        if (err) throw err;
        res.type("png");
        if (rows[0]["img"] != null) res.send(rows[0]["img"]);
      }
    );
  });

  app.get("/correctionimg5", function (req, res) {
    var conn = mysql.createConnection({
      host: "210.125.31.236",
      port: "3306",
      user: "test2",
      password: "s1234",
      database: "upright",
    });
    conn.connect();
    var query = conn.query(
      "SELECT img FROM imglist where id='" +
        id +
        "' order by number desc LIMIT 4, 1",
      function (err, rows) {
        if (err) throw err;
        res.type("png");
        if (rows[0]["img"] != null) res.send(rows[0]["img"]);
      }
    );
  });

  app.get("/correctionimg6", function (req, res) {
    var conn = mysql.createConnection({
      host: "210.125.31.236",
      port: "3306",
      user: "test2",
      password: "s1234",
      database: "upright",
    });
    conn.connect();
    var query = conn.query(
      "SELECT img FROM imglist where id='" +
        id +
        "' order by number desc LIMIT 5, 1",
      function (err, rows) {
        if (err) throw err;
        res.type("png");
        if (rows[0]["img"] != null) res.send(rows[0]["img"]);
      }
    );
  });

  app.get("/correctionimg7", function (req, res) {
    var conn = mysql.createConnection({
      host: "210.125.31.236",
      port: "3306",
      user: "test2",
      password: "s1234",
      database: "upright",
    });
    conn.connect();
    var query = conn.query(
      "SELECT img FROM imglist where id='" +
        id +
        "' order by number desc LIMIT 6, 1",
      function (err, rows) {
        if (err) throw err;
        res.type("png");
        if (rows[0]["img"] != null) res.send(rows[0]["img"]);
      }
    );
  });

  app.get("/correctionimg8", function (req, res) {
    var conn = mysql.createConnection({
      host: "210.125.31.236",
      port: "3306",
      user: "test2",
      password: "s1234",
      database: "upright",
    });
    conn.connect();
    var query = conn.query(
      "SELECT img FROM imglist where id='" +
        id +
        "' order by number desc LIMIT 7, 1",
      function (err, rows) {
        if (err) throw err;
        res.type("png");
        if (rows[0]["img"] != null) res.send(rows[0]["img"]);
      }
    );
  });

  app.get("/correctionimg9", function (req, res) {
    var conn = mysql.createConnection({
      host: "210.125.31.236",
      port: "3306",
      user: "test2",
      password: "s1234",
      database: "upright",
    });
    conn.connect();
    var query = conn.query(
      "SELECT img FROM imglist where id='" +
        id +
        "' order by number desc LIMIT 8, 1",
      function (err, rows) {
        if (err) throw err;
        res.type("png");
        if (rows[0]["img"] != null) res.send(rows[0]["img"]);
      }
    );
  });

  app.get("/correctionimg10", function (req, res) {
    var conn = mysql.createConnection({
      host: "210.125.31.236",
      port: "3306",
      user: "test2",
      password: "s1234",
      database: "upright",
    });
    conn.connect();
    var query = conn.query(
      "SELECT img FROM imglist where id='" +
        id +
        "' order by number desc LIMIT 9, 1",
      function (err, rows) {
        if (err) throw err;
        res.type("png");
        if (rows[0]["img"] != null) res.send(rows[0]["img"]);
      }
    );
  });

  app.get("/correctionimg11", function (req, res) {
    var conn = mysql.createConnection({
      host: "210.125.31.236",
      port: "3306",
      user: "test2",
      password: "s1234",
      database: "upright",
    });
    conn.connect();
    var query = conn.query(
      "SELECT img FROM imglist where id='" +
        id +
        "' order by number desc LIMIT 10, 1",
      function (err, rows) {
        if (err) throw err;
        res.type("png");
        if (rows[0]["img"] != null) res.send(rows[0]["img"]);
      }
    );
  });

  app.get("/correctionimg12", function (req, res) {
    var conn = mysql.createConnection({
      host: "210.125.31.236",
      port: "3306",
      user: "test2",
      password: "s1234",
      database: "upright",
    });
    conn.connect();
    var query = conn.query(
      "SELECT img FROM imglist where id='" +
        id +
        "' order by number desc LIMIT 11, 1",
      function (err, rows) {
        if (err) throw err;
        res.type("png");
        if (rows[0]["img"] != null) res.send(rows[0]["img"]);
      }
    );
  });
  process.on("uncaughtException", (req, res, err) => {
    console.error("죽지마 ㅠㅠ");
    console.error(err);
    app.use(function (req, res) {
      res.status(500).send("서버오류가 발생하였습니다.");
    });
  });
  /*
  app.use(function (err, req, res, next) {
    console.error("서버 오류 발생", err);
    res.status(500).send('서버오류가 발생하였습니다.');
  });
  */
});
