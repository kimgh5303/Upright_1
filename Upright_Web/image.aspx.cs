namespace imgTestWeb{
    using System;
    using System.Date;
    using System.Data.SqlClient;

    public partial class image : System.Web.UI.Page
    {
        protected void Page_Load(object sender, EventArgs e)
        {
            string idString = Request["Id"].ToSring();
            int id = int.Parse(idString);

            DataSet ds = LoadData(id);
            DataRow row = ds.Tables[0].Rows[0];

            byte[] imgBytes = (byte[]) row["Img"];
            string imgType = row["ImgType"].ToString;

            Response.Clear();
            Response.ContentType = "image/" + imgType;
            Response.AppendHeader("Content-Length", imgBytes.Length.ToString());
            Response.AppendHeader("Pragma","public");
            Response.BinaryWrite(imgBytes);
            Response.Flush();
            Response.Close();
        }
        private DataSet LoadData(int id)
        {
            string strConn = "Data Source = .;Initial Catalog = MyDG;Integrated Security=SSPI;";
            DataSet ds = new DataSet();

            using (SqlConnection conn = new SqlConnection(strConn))        
            {
                conn.Open();

                string sql = "SELECT Img, ImgType FROM ImgTable WHERE Id=" + id.ToString();
                SqlDataAdapter adapter = new SqlDataAdapter(sql, conn);
                adapter.Fill(ds);
            }
            return ds;
        }
    }
}


private static Dictionary imageHeaderFormat = new Dictionary()
{
	{new byte[] {0x42, 0x4D}, ImageFormat.Bmp},
	{new byte[] {0x47, 0x49, 0x46, 0x38, 0x37, 0x61}, ImageFormat.Gif},
	{new byte[] {0x47, 0x49, 0x46, 0x38, 0x39, 0x61}, ImageFormat.Gif},
	{new byte[] {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}, ImageFormat.Png},
	{new byte[] {0xff, 0xd8}, ImageFormat.Jpeg}
};