const mode = document.getElementById("imgDiv");

function hihi() {
  var oReq = new XMLHttpRequest();

  oReq.open("POST", "/imgCnt");
  oReq.setRequestHeader("Content-Type", "application/json"); // json 형태로 보낸다
  oReq.send();

  oReq.addEventListener("load", function () {
    var result = JSON.parse(oReq.responseText);
    console.log(result);
    if (result > 12) result = 12;
    update(result);
  });
}

function update(e) {
  for (let i = 0; i < e; i++) {
    var xhr = new XMLHttpRequest();
    let url = "/correctionimg" + (i + 1);
    xhr.open("GET", url);
    xhr.responseType = "blob";
    xhr.onload = response;
    xhr.send();

    function response() {
      var urlCreator = window.URL || window.webkitURL;
      var imageUrl = urlCreator.createObjectURL(this.response);
      //document.querySelector("#image").src = imageUrl;

      var new_img = document.createElement("img");
      new_img.setAttribute("src", imageUrl);
      new_img.setAttribute("id", "imgEl");
      mode.appendChild(new_img);
    }
  }
}
